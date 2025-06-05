import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

from models.SwinTransformerVit import SwinTransformerVit
from losses.CombinedLoss import CombinedLoss
from dataset.dataloader import CarvanaDataset

# ---- Segmentation Metrics ----
def dice_coefficient(y_pred, y_true, threshold=0.5):
    smooth = 1e-6
    y_pred = (torch.sigmoid(y_pred) > threshold).float()
    y_true = (y_true > threshold).float()
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return min(dice.item(), 1.0)

def jaccard_coefficient(y_pred, y_true, threshold=0.5):
    smooth = 1e-6
    y_pred = (torch.sigmoid(y_pred) > threshold).float()
    y_true = (y_true > threshold).float()
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    return jaccard.item()

def accuracy(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred) > 0.5
    correct = (y_pred == y_true.byte()).float().sum()
    total = y_true.numel()
    return (correct / total).item()

def precision(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred) > 0.5
    true_positives = torch.sum(y_pred * y_true.byte())
    predicted_positives = torch.sum(y_pred)
    return (true_positives / predicted_positives).item() if predicted_positives else 0

def recall(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred) > 0.5
    true_positives = torch.sum(y_pred * y_true.byte())
    actual_positives = torch.sum(y_true)
    return (true_positives / actual_positives).item() if actual_positives else 0

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 100
WEIGHT_DECAY = 1e-4
DATA_PATH = "src/segmentation_model/dataset/data"
MODEL_SAVE_PATH = "./saved_models/logsage_cbam_best.pth"

# Setup
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset
dataset = CarvanaDataset(DATA_PATH)
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model, Loss, Optimizer
model = SwinTransformerVit(in_channels=1, out_channels=1).to(device)
criterion = CombinedLoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
scaler = GradScaler()

# Tracking
train_losses, val_losses = [], []
best_val_loss = float('inf')
best_metrics = {}

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_metrics = {'acc': 0, 'dice': 0, 'jaccard': 0, 'prec': 0, 'rec': 0}

    for img, dog, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        img, dog, mask = img.to(device), dog.to(device), mask.to(device)
        optimizer.zero_grad()
        with autocast():
            y_pred = model(img, dog)
            loss = criterion(y_pred, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()
        train_metrics['acc'] += accuracy(y_pred, mask)
        train_metrics['dice'] += dice_coefficient(y_pred, mask)
        train_metrics['jaccard'] += jaccard_coefficient(y_pred, mask)
        train_metrics['prec'] += precision(y_pred, mask)
        train_metrics['rec'] += recall(y_pred, mask)

    n_train = len(train_loader)
    train_loss /= n_train
    train_losses.append(train_loss)
    for k in train_metrics:
        train_metrics[k] /= n_train
    train_f1 = f1_score(train_metrics['prec'], train_metrics['rec'])

    # Validation
    model.eval()
    val_loss = 0
    val_metrics = {'acc': 0, 'dice': 0, 'jaccard': 0, 'prec': 0, 'rec': 0}

    with torch.no_grad():
        for img, dog, mask in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            img, dog, mask = img.to(device), dog.to(device), mask.to(device)
            y_pred = model(img, dog)
            loss = criterion(y_pred, mask)
            val_loss += loss.item()
            val_metrics['acc'] += accuracy(y_pred, mask)
            val_metrics['dice'] += dice_coefficient(y_pred, mask)
            val_metrics['jaccard'] += jaccard_coefficient(y_pred, mask)
            val_metrics['prec'] += precision(y_pred, mask)
            val_metrics['rec'] += recall(y_pred, mask)

    n_val = len(val_loader)
    val_loss /= n_val
    val_losses.append(val_loss)
    for k in val_metrics:
        val_metrics[k] /= n_val
    val_f1 = f1_score(val_metrics['prec'], val_metrics['rec'])

    scheduler.step(val_loss)

    print(f"\n[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Train: Dice={train_metrics['dice']:.4f}, Jaccard={train_metrics['jaccard']:.4f}, Acc={train_metrics['acc']:.4f}, F1={train_f1:.4f}")
    print(f"Val:   Dice={val_metrics['dice']:.4f}, Jaccard={val_metrics['jaccard']:.4f}, Acc={val_metrics['acc']:.4f}, F1={val_f1:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        best_metrics = {
            'epoch': epoch + 1,
            'dice': val_metrics['dice'],
            'jaccard': val_metrics['jaccard'],
            'acc': val_metrics['acc'],
            'f1': val_f1
        }
        print(f"\n✅ Best model saved at epoch {epoch+1} with Val Loss: {val_loss:.4f}")

# Plot Losses
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(MODEL_SAVE_PATH), "loss_curve.png"), dpi=300)
plt.close()

# Final Stats
print("\nBest Model Performance:")
for k, v in best_metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
