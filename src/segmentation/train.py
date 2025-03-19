import os
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from tqdm import tqdm
import monai
from unet3d import UNet3D  # Import your UNet3D model
from dataset_loader import Dataset  # Custom dataset loader

# Define metrics
def accuracy(y_pred, y_true):
    y_pred = torch.sigmoid(y_pred) > 0.5
    correct = (y_pred == y_true.byte()).float().sum()
    total = y_true.numel()
    return (correct / total).item()

def dice_coefficient(y_pred, y_true):
    smooth = 1e-6
    y_pred = torch.sigmoid(y_pred)
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def jaccard_coefficient(y_pred, y_true):
    smooth = 1e-6
    y_pred = torch.sigmoid(y_pred)
    intersection = torch.sum(y_pred * y_true)
    union = torch.sum(y_pred) + torch.sum(y_true) - intersection
    jaccard = (intersection + smooth) / (union + smooth)
    return jaccard.item()

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

# Training function
def train_segmentation_model(
    data_path,
    save_path,
    epochs=50,
    batch_size=1,
    learning_rate=1e-4
):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data directory '{data_path}' does not exist.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = Dataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = UNet3D(in_channels=1, out_channels=1).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')

    for epoch in range(epochs):
        model.train()
        train_loss, train_metrics = run_epoch(train_loader, model, criterion, optimizer, device, is_train=True)
        model.eval()
        val_loss, val_metrics = run_epoch(val_loader, model, criterion, optimizer, device, is_train=False)

        print("-" * 30)
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, {train_metrics}")
        print(f"Validation Loss: {val_loss:.4f}, {val_metrics}")
        print("-" * 30)

        torch.save(model.state_dict(), os.path.join(save_path, f"unet3d_epoch{epoch + 1}.pth"))
        print(f"Model saved to {save_path}")

def run_epoch(loader, model, criterion, optimizer, device, is_train=True):
    epoch_loss = 0
    metrics = {
        "accuracy": 0,
        "dice": 0,
        "jaccard": 0,
        "precision": 0,
        "recall": 0,
        "f1": 0
    }
    for images, masks in tqdm(loader, desc="Training" if is_train else "Validation"):
        images, masks = images.to(device), masks.to(device)

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)
        epoch_loss += loss.item()

        metrics["accuracy"] += accuracy(outputs, masks)
        metrics["dice"] += dice_coefficient(outputs, masks)
        metrics["jaccard"] += jaccard_coefficient(outputs, masks)
        metrics["precision"] += precision(outputs, masks)
        metrics["recall"] += recall(outputs, masks)

        if is_train:
            loss.backward()
            optimizer.step()

    dataset_len = len(loader)
    metrics = {k: v / dataset_len for k, v in metrics.items()}
    metrics["f1"] = f1_score(metrics["precision"], metrics["recall"])

    return epoch_loss / dataset_len, metrics

if __name__ == "__main__":
    DATA_PATH = "/home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/data"
    SAVE_PATH = "/home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/saved_model"

    os.makedirs(SAVE_PATH, exist_ok=True)

    train_segmentation_model(
        data_path=DATA_PATH,
        save_path=SAVE_PATH,
        epochs=50,
        batch_size=1,
        learning_rate=1e-4
    )
