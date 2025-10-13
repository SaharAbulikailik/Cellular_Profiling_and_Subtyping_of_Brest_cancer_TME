#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LoGSAGE-CBAM inference on .czi + watershed instances

Usage (example):
  export PYTHONPATH="$PWD/src:$PYTHONPATH"
  python src/analysis/Generate_masks.py \
    --model /home/sahar/cellscopes/src/segmentation_model/saved_models/LoGSAGE_Multispec_sigma_Fusion3.pth \
    --images /home/sahar/cellscopes/src/analysis/Test_images \
    --out-masks /home/sahar/cellscopes/src/analysis/Test_images/LoGSAGE-CBAM_masks \
    --out-labels /home/sahar/cellscopes/src/analysis/Test_images/LoGSAGE-CBAM_labels \
    --thresh 0.5 \
    --min-distance 9
"""

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

# Post-processing
from scipy.ndimage import distance_transform_edt
from skimage import morphology, segmentation, measure
from skimage.feature import peak_local_max
import tifffile as tiff

# ==== import your model pieces (ensure PYTHONPATH includes repo/src) ====
from segmentation_model.models.swin_T_1 import SwinTransformer
from segmentation_model.models.Fusion import CBAMFusionBlock


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

# ----------------- helpers -----------------
def load_model(model_path: str, device: torch.device):
    # Support checkpoints saved with or without weights_only
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    # infer saliency channels
    key = 'encoder1.conv.0.weight'
    sal_ch = state[key].shape[1] if key in state else 1
    print(f"[info] checkpoint expects saliency channels = {sal_ch}")
    model = SwinTransformerVit(in_channels=sal_ch, out_channels=1).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, sal_ch

def make_dog_single_channel(patch_uint8):
    ch0 = patch_uint8[..., 0].astype(np.float32) / 255.0
    dog = difference_of_gaussians(ch0, 8)
    mn, mx = dog.min(), dog.max()
    dog = (dog - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(dog, dtype=np.float32)
    return dog[..., None].astype(np.float32)

def make_dog_from_first_channel(patch_uint8):
    ch0 = patch_uint8[..., 0].astype(np.float32) / 255.0
    dog = difference_of_gaussians(ch0, 8)
    mn, mx = dog.min(), dog.max()
    dog = (dog - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(dog, dtype=np.float32)
    return np.repeat(dog[..., None], 3, axis=-1).astype(np.float32)

def predict_and_reassemble_czi(patches, model, device, sal_ch, thresh=0.5):
    """
    patches: (nrows, ncols, 1, 256,256,3) uint8
    returns (H,W) uint8 binary mask in {0,1}
    """
    nrows, ncols = patches.shape[0], patches.shape[1]
    core = 128
    cores = np.zeros((nrows, ncols, core, core), dtype=np.uint8)

    with torch.no_grad():
        for i in range(nrows):
            for j in range(ncols):
                patch = patches[i, j, 0]  # (256,256,3)
                img_np = patch.astype(np.float32) / 255.0
                dog_np = make_dog_single_channel(patch) if sal_ch == 1 else make_dog_from_first_channel(patch)
                t_img = torch.from_numpy(img_np.transpose(2,0,1))[None].float().to(device)
                t_dog = torch.from_numpy(dog_np.transpose(2,0,1))[None].float().to(device)

                logits = model(t_img, t_dog)
                prob = torch.sigmoid(logits)[0,0].cpu().numpy()
                pred = (prob > thresh).astype(np.uint8)

                off = (256 - core)//2
                cores[i, j] = pred[off:off+core, off:off+core]

    H = nrows * core
    W = ncols * core
    out = np.zeros((H, W), dtype=np.uint8)
    for i in range(nrows):
        for j in range(ncols):
            y0, x0 = i * core, j * core
            out[y0:y0+core, x0:x0+core] = cores[i, j]
    return out

def watershed_from_pred(pred_mask: np.ndarray, min_distance: int = 9) -> np.ndarray:
    """
    EXACT method you requested:
      distance = distance_transform_edt(pred_mask)
      local_max_coords = peak_local_max(distance, labels=pred_mask, min_distance=..., exclude_border=False)
      markers = measure.label(local_max_mask)
      labels = segmentation.watershed(-distance, markers, mask=pred_mask)
    """
    pred_mask = (pred_mask > 0)
    if not pred_mask.any():
        return np.zeros_like(pred_mask, dtype=np.int32)

    pred_mask = morphology.remove_small_objects(pred_mask, min_size=40)
    distance = distance_transform_edt(pred_mask)
    coords = peak_local_max(distance, labels=pred_mask.astype(np.uint8),
                            min_distance=min_distance, exclude_border=False)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    if coords.size > 0:
        local_max_mask[tuple(coords.T)] = True

    markers = measure.label(local_max_mask)
    if markers.max() == 0:
        markers = measure.label(pred_mask.astype(np.uint8))
    labels = segmentation.watershed(-distance, markers=markers, mask=pred_mask)
    return labels.astype(np.int32)

# ----------------- Runner -----------------
def process_images_czi(images_dir: Path, out_masks: Path, out_labels: Path,
                       model_path: Path, prob_thresh=0.5, min_distance=9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sal_ch = load_model(str(model_path), device)

    out_masks.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(".czi")])
    if not files:
        print(f"[warn] No .czi files in {images_dir}")
        return

    for fname in files:
        fp = images_dir / fname
        print(f"[info] Processing {fname}")
        # read CZI: (C,Y,X,1) -> make uint8 RGB (first 3 channels)
        arr = czifile.imread(str(fp))
        I0 = cv2.normalize(arr[0,:,:,0], None, 0, 255, cv2.NORM_MINMAX)
        I1 = cv2.normalize(arr[1,:,:,0], None, 0, 255, cv2.NORM_MINMAX)
        I2 = cv2.normalize(arr[2,:,:,0], None, 0, 255, cv2.NORM_MINMAX)
        rgb = np.stack((I0, I1, I2), axis=2).astype(np.uint8)

        # pad + tile
        imagein = np.pad(rgb, ((64,128), (64,64), (0,0)), mode='constant')
        patches = patchify(imagein, (256, 256, 3), step=(128, 128, 3))

        # predict + stitch
        mask_bin = predict_and_reassemble_czi(patches, model, device, sal_ch, thresh=prob_thresh)

        # watershed instances
        labels = watershed_from_pred(mask_bin, min_distance=min_distance)

        base = fname[:-4]  # strip ".czi"
        cv2.imwrite(str(out_masks / f"{base}_mask.png"), (mask_bin * 255).astype(np.uint8))
        tiff.imwrite(str(out_labels / f"{base}_labels.tiff"), labels.astype(np.uint16), compression='zlib')
        print(f"[ok] Saved mask   -> {out_masks / f'{base}_mask.png'}")
        print(f"[ok] Saved labels -> {out_labels / f'{base}_labels.tiff'}")

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="LoGSAGE-CBAM inference + watershed instances")
    p.add_argument("--model", required=True, help="Path to .pth weights")
    p.add_argument("--images", required=True, help="Folder with .czi files")
    p.add_argument("--out-masks", required=True, help="Output folder for binary masks (PNG)")
    p.add_argument("--out-labels", required=True, help="Output folder for instance labels (TIFF)")
    p.add_argument("--thresh", type=float, default=0.5, help="Probability threshold for binarization")
    p.add_argument("--min-distance", type=int, default=9, help="peak_local_max min_distance")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_images_czi(
        images_dir=Path(args.images),
        out_masks=Path(args.out_masks),
        out_labels=Path(args.out_labels),
        model_path=Path(args.model),
        prob_thresh=args.thresh,
        min_distance=args.min_distance
    )
