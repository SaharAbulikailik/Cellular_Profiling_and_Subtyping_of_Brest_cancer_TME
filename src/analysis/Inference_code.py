import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import cv2
import czifile
from pathlib import Path
from patchify import patchify
from skimage.filters import difference_of_gaussians
from skimage.color import label2rgb

# Post-processing
from scipy.ndimage import distance_transform_edt
from skimage import morphology, segmentation, measure
from skimage.feature import peak_local_max
import tifffile as tiff

# ==== import your model pieces (ensure PYTHONPATH includes repo/src) ====
from swin_T_1 import SwinTransformer
from Fusion import CBAMFusionBlock


# ----------------- Blocks -----------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(out_channels * 2, out_channels)  # concat with skip
    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        return self.double_conv(x)

# ----------------- Swin + CBAM fusion model -----------------
class SwinTransformerVit(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.swinViT = SwinTransformer()

        # image path feature adapters
        self.conv_in_img = nn.Sequential(nn.Conv2d(3, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.conv1_img = nn.Sequential(nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.conv2_img = nn.Sequential(nn.Conv2d(96, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.conv3_img = nn.Sequential(nn.Conv2d(192,192,3, padding=1), nn.BatchNorm2d(192),nn.ReLU(inplace=True))
        self.conv4_img = nn.Sequential(nn.Conv2d(384,384,3, padding=1), nn.BatchNorm2d(384),nn.ReLU(inplace=True))

        # saliency encoder
        self.encoder1 = DoubleConv(in_channels, 48)
        self.encoder2 = DoubleConv(48, 96)
        self.encoder3 = DoubleConv(96, 192)
        self.encoder4 = DoubleConv(192, 384)
        self.maxpool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.4)

        # CBAM fusion
        self.fuse1 = CBAMFusionBlock(48, 48, 96)
        self.fuse2 = CBAMFusionBlock(96, 96, 192)
        self.fuse3 = CBAMFusionBlock(192, 192, 384)
        self.fuse4 = CBAMFusionBlock(384, 384, 768)

        # decoder
        self.decoder1 = DecoderBlock(768, 384)
        self.decoder2 = DecoderBlock(384, 192)
        self.decoder3 = DecoderBlock(192, 96)
        self.decoder4 = DecoderBlock(96, 48)
        self.final_conv = nn.Conv2d(48, out_channels, kernel_size=1)

    def forward(self, x, saliency):
        # image features via Swin
        skip0 = self.conv_in_img(x)
        s1, s2, s3, s4 = self.swinViT(x)
        s1 = self.conv1_img(s1)
        s2 = self.conv2_img(s2)
        s3 = self.conv3_img(s3)
        s4 = self.conv4_img(s4)

        # saliency encoder
        xs = self.encoder1(saliency); xs = self.maxpool(xs); xs = self.dropout(xs);  s1s = xs
        xs = self.encoder2(xs);       xs = self.maxpool(xs); xs = self.dropout(xs);  s2s = xs
        xs = self.encoder3(xs);       xs = self.maxpool(xs); xs = self.dropout(xs);  s3s = xs
        xs = self.encoder4(xs);       xs = self.maxpool(xs); xs = self.dropout(xs);  s4s = xs

        # CBAM fuse
        F1 = self.fuse1(s1, s1s)
        F2 = self.fuse2(s2, s2s)
        F3 = self.fuse3(s3, s3s)
        F4 = self.fuse4(s4, s4s)

        # decode
        y = self.decoder1(F4, F3)
        y = self.decoder2(y, F2)
        y = self.decoder3(y, F1)
        y = self.decoder4(y, skip0)
        return self.final_conv(y)

# --- Helpers ---
def make_dog(image):
    ch0 = image.astype(np.float32) / 255.0
    dog = difference_of_gaussians(ch0, 8)
    mn, mx = dog.min(), dog.max()
    return ((dog - mn) / (mx - mn + 1e-8)).astype(np.float32)

def watershed_from_pred(pred_mask: np.ndarray, min_distance: int = 9) -> np.ndarray:
    pred_mask = (pred_mask > 0)
    if not pred_mask.any():
        return np.zeros_like(pred_mask, dtype=np.int32)

    pred_mask = morphology.remove_small_objects(pred_mask, min_size=40)
    distance = distance_transform_edt(pred_mask)
    coords = peak_local_max(distance, labels=pred_mask, min_distance=min_distance, exclude_border=False)
    local_max_mask = np.zeros_like(pred_mask, dtype=bool)
    if coords.size > 0:
        local_max_mask[tuple(coords.T)] = True
    markers = measure.label(local_max_mask)
    if markers.max() == 0:
        markers = measure.label(pred_mask.astype(np.uint8))
    labels = segmentation.watershed(-distance, markers, mask=pred_mask)
    return labels.astype(np.int32)

def load_model(model_path, device):
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    key = 'encoder1.conv.0.weight'
    sal_ch = state[key].shape[1] if key in state else 1
    model = SwinTransformerVit(in_channels=sal_ch, out_channels=1).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, sal_ch

# --- Inference ---
def run_inference_on_folder(img_dir, out_masks, out_labels, model_path, threshold=0.5, min_distance=9, out_overlay=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sal_ch = load_model(model_path, device)

    os.makedirs(out_masks, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)
    if out_overlay:
        os.makedirs(out_overlay, exist_ok=True)

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue
        img_path = os.path.join(img_dir, fname)
        print(f"[INFO] Processing {fname}")

        # --- Load grayscale image ---
        img = tiff.imread(img_path)
        if img.ndim == 3:
            img = img[0]
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # --- Model input ---
        img_rgb = np.stack([img, img, img], axis=-1)
        t_img = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device) / 255.0

        dog = make_dog(img)
        if sal_ch == 1:
            t_dog = torch.from_numpy(dog).unsqueeze(0).unsqueeze(0).to(device)
        else:
            dog3 = np.stack([dog, dog, dog], axis=-1)
            t_dog = torch.from_numpy(dog3.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

        # --- Inference ---
        with torch.no_grad():
            logits = model(t_img, t_dog)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_bin = (prob > threshold).astype(np.uint8)

        # --- Watershed ---
        labels = watershed_from_pred(pred_bin, min_distance=min_distance)

        # --- Save masks and labels ---
        base = os.path.splitext(fname)[0]
        cv2.imwrite(os.path.join(out_masks, f"{base}_mask.png"), pred_bin * 255)
        tiff.imwrite(os.path.join(out_labels, f"{base}_labels.tiff"), labels.astype(np.uint16), compression='zlib')

        # --- Overlay (optional) ---
        if out_overlay:
            image_float = img.astype(np.float32) / 255.0
            image_rgb = np.stack([image_float]*3, axis=-1)
            overlay = label2rgb(labels, image=image_rgb, bg_label=0, alpha=0.4, saturation=1.0)
            overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_overlay, f"{base}_overlay.png"), cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR))

        print(f"[OK] Saved: {base}_mask.png, {base}_labels.tiff" + (f", {base}_overlay.png" if out_overlay else ""))

# --- Entry ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--out-masks", required=True)
    parser.add_argument("--out-labels", required=True)
    parser.add_argument("--thresh", type=float, default=0.5)
    parser.add_argument("--min-distance", type=int, default=9)
    parser.add_argument("--out-overlay", required=True, help="Output folder for overlay visualizations (PNG)")
    args = parser.parse_args()

    run_inference_on_folder(
        img_dir=args.images,
        out_masks=args.out_masks,
        out_labels=args.out_labels,
        model_path=args.model,
        threshold=args.thresh,
        min_distance=args.min_distance,
        out_overlay=args.out_overlay
    )
