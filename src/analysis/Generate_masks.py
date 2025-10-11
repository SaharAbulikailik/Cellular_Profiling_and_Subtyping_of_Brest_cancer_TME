#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- CZI -> LoGSAGE-CBAM inference + watershed instances (your exact method) ---

import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import czifile
from patchify import patchify
from skimage.filters import difference_of_gaussians

# Post-processing
from scipy.ndimage import distance_transform_edt
from skimage import morphology, segmentation, measure
from skimage.feature import peak_local_max
from skimage.morphology import disk
import tifffile as tiff

# ==== import your model pieces (ensure these modules are on PYTHONPATH) ====
from segmentation_model.models.swin_T_1 import SwinTransformer
from segmentation_model.models.Fusion import CBAMFusionBlock


# ----------------- Model (TransUNet backbone + CBAM fusion) -----------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
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
        super(DecoderBlock, self).__init__()
        # The in_channels for upconv is adjusted to out_channels to match the skip connection size.
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # Adjust double_conv to take in_channels (from upconv and skip connection) and out_channels.
        # The input channels to double_conv is now out_channels * 2, as it includes concatenated skip connection.
        self.double_conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        # Ensure that the concatenation is between the upsampled feature map and the skip connection.
        x = torch.cat([x, skip], dim=1)  # Concatenate on the channel dimension
        x = self.double_conv(x)
        return x
    
# Swin Transformer-based model with defined input and output channels
class SwinTransformerVit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SwinTransformerVit, self).__init__()
        self.swinViT = SwinTransformer()
        
        # Encoder for the saliency map with batch norm included in DoubleConv
        self.encoder1 = DoubleConv(in_channels, 48)
        self.encoder2 = DoubleConv(48, 96)
        self.encoder3 = DoubleConv(96, 192)
        self.encoder4 = DoubleConv(192, 384)

        # MaxPooling and Dropout layers applied globally
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.4)

        
        
        self.conv_in_img = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.conv1_img = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.conv2_img = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.conv3_img = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.conv4_img = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))


        # BiFusion blocks instead of concatenation
        self.fuse1 = CBAMFusionBlock(48, 48, 96)
        self.fuse2 = CBAMFusionBlock(96, 96, 192)
        self.fuse3 = CBAMFusionBlock(192, 192, 384)
        self.fuse4 = CBAMFusionBlock(384, 384, 768)



        self.decoder1 = DecoderBlock(768, 384)
        self.decoder2 = DecoderBlock(384, 192)
        self.decoder3 = DecoderBlock(192, 96)
        self.decoder4 = DecoderBlock(96, 48)

        self.final_conv = nn.Conv2d(48, 1, kernel_size=1)


    def forward(self, x, saliency):
        # print(f"Input x: {x.shape}, saliency: {saliency.shape}")

        # Swin feature extraction
        skip0 = self.conv_in_img(x)
        # print(f"skip0 (after conv_in_img): {skip0.shape}")

        skip1, skip2, skip3, skip4 = self.swinViT(x)
        # print(f"skip1 from swinViT: {skip1.shape}")
        # print(f"skip2 from swinViT: {skip2.shape}")
        # print(f"skip3 from swinViT: {skip3.shape}")
        # print(f"skip4 from swinViT: {skip4.shape}")

        skip1 = self.conv1_img(skip1)
        skip2 = self.conv2_img(skip2)
        skip3 = self.conv3_img(skip3)
        skip4 = self.conv4_img(skip4)

        # print(f"skip1 after conv1_img: {skip1.shape}")
        # print(f"skip2 after conv2_img: {skip2.shape}")
        # print(f"skip3 after conv3_img: {skip3.shape}")
        # print(f"skip4 after conv4_img: {skip4.shape}")

        # Saliency encoder
        x_sal = self.encoder1(saliency)
        x_sal = self.maxpool(x_sal)
        x_sal = self.dropout(x_sal)
        skip1_sal = x_sal
        # print(f"skip1_sal: {skip1_sal.shape}")

        x_sal = self.encoder2(x_sal)
        x_sal = self.maxpool(x_sal)
        x_sal = self.dropout(x_sal)
        skip2_sal = x_sal
        # print(f"skip2_sal: {skip2_sal.shape}")

        x_sal = self.encoder3(x_sal)
        x_sal = self.maxpool(x_sal)
        x_sal = self.dropout(x_sal)
        skip3_sal = x_sal
        # print(f"skip3_sal: {skip3_sal.shape}")

        x_sal = self.encoder4(x_sal)
        x_sal = self.maxpool(x_sal)
        x_sal = self.dropout(x_sal)
        skip4_sal = x_sal
        # print(f"skip4_sal: {skip4_sal.shape}")

        # Attention Fusion
        F_Skip1 = self.fuse1(skip1, skip1_sal)
        # print(f"F_Skip1: {F_Skip1.shape}")
        F_Skip2 = self.fuse2(skip2, skip2_sal)
        # print(f"F_Skip2: {F_Skip2.shape}")
        F_Skip3 = self.fuse3(skip3, skip3_sal)
        # print(f"F_Skip3: {F_Skip3.shape}")
        F_Skip4 = self.fuse4(skip4, skip4_sal)
        # print(f"F_Skip4: {F_Skip4.shape}")

        # Decoder
        x = self.decoder1(F_Skip4, F_Skip3)
        # print(f"After decoder1: {x.shape}")
        x = self.decoder2(x, F_Skip2)
        # print(f"After decoder2: {x.shape}")
        x = self.decoder3(x, F_Skip1)
        # print(f"After decoder3: {x.shape}")
        x = self.decoder4(x, skip0)
        # print(f"After decoder4: {x.shape}")

        out = self.final_conv(x)
        # print(f"Final output: {out.shape}")
        return out

# ----------------- Load checkpoint -----------------
def load_model(model_path, device):
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    key = 'encoder1.conv.0.weight'
    sal_ch = state[key].shape[1] if key in state else 1  # 1 or 3
    print(f"[info] checkpoint expects saliency channels = {sal_ch}")
    model = SwinTransformerVit(in_channels=sal_ch, out_channels=1).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, sal_ch


# ----------------- Saliency (LoG/DoG) helpers -----------------
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


# ----------------- Sliding-window predict + stitch -----------------
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
                patch = patches[i, j, 0]  # (256,256,3) uint8
                img_np = patch.astype(np.float32) / 255.0

                if sal_ch == 1:
                    dog_np = make_dog_single_channel(patch)
                else:
                    dog_np = make_dog_from_first_channel(patch)

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


# ----------------- Your exact watershed post-processing -----------------
def watershed_from_pred(pred_mask: np.ndarray, min_distance: int = 9) -> np.ndarray:
    """
    pred_mask: (H,W) uint8/bool binary mask
    1) distance_transform_edt
    2) peak_local_max(labels=pred_mask, min_distance=..., exclude_border=False)
    3) measure.label -> markers
    4) segmentation.watershed(-distance, markers, mask=pred_mask)
    Returns int32 labels (0=background).
    """
    pred_mask = (pred_mask > 0)
    if not pred_mask.any():
        return np.zeros_like(pred_mask, dtype=np.int32)

    # Optional: light tidy to reduce spurious seeds
    pred_mask = morphology.remove_small_objects(pred_mask, min_size=40)

    distance = distance_transform_edt(pred_mask)
    coords = peak_local_max(
        distance,
        labels=pred_mask.astype(np.uint8),
        min_distance=min_distance,
        exclude_border=False
    )
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    if coords.size > 0:
        local_max_mask[tuple(coords.T)] = True

    markers = measure.label(local_max_mask)
    if markers.max() == 0:
        markers = measure.label(pred_mask.astype(np.uint8))

    labels = segmentation.watershed(-distance, markers=markers, mask=pred_mask)
    return labels.astype(np.int32)


# ----------------- Main processing loop -----------------
def process_images_czi(input_folder, output_folder, model_path,
                       prob_thresh=0.5, min_distance=9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sal_ch = load_model(model_path, device)

    images_folder = os.path.join(input_folder, 'Test_images')  # <-- set to your images subfolder
    masks_folder  = os.path.join(output_folder, 'LoGSAGE-CBAM_masks')
    labels_folder = os.path.join(output_folder, 'LoGSAGE-CBAM_labels')
    os.makedirs(masks_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    for filename in os.listdir(images_folder):
        if not filename.lower().endswith('.czi'):
            continue

        file_path = os.path.join(images_folder, filename)
        print(f"[info] Processing {filename}")

        # read CZI (expects (C,Y,X,1)) -> make 3ch uint8
        imageorg = czifile.imread(file_path)
        I11 = imageorg[0, :, :, 0]
        I21 = imageorg[1, :, :, 0]
        I31 = imageorg[2, :, :, 0]

        I11 = cv2.normalize(I11, None, 0, 255, cv2.NORM_MINMAX)
        I21 = cv2.normalize(I21, None, 0, 255, cv2.NORM_MINMAX)
        I31 = cv2.normalize(I31, None, 0, 255, cv2.NORM_MINMAX)
        rgb_image1 = np.stack((I11, I21, I31), axis=2).astype(np.uint8)

        # pad + tile
        imagein = np.pad(rgb_image1, ((64,128), (64,64), (0,0)), mode='constant')
        patches = patchify(imagein, (256, 256, 3), step=(128, 128, 3))

        # predict + stitch -> binary mask
        mask_bin = predict_and_reassemble_czi(patches, model, device, sal_ch=sal_ch, thresh=prob_thresh)

        # your watershed method
        labels = watershed_from_pred(mask_bin, min_distance=min_distance)

        # save
        base = filename.replace('.czi','')
        mask_png_path   = os.path.join(masks_folder,  f"{base}_mask.png")
        labels_tif_path = os.path.join(labels_folder, f"{base}_labels.tiff")

        cv2.imwrite(mask_png_path, (mask_bin * 255).astype(np.uint8))
        tiff.imwrite(labels_tif_path, labels.astype(np.uint16), compression='zlib')

        print(f"[ok] Saved binary:   {mask_png_path}")
        print(f"[ok] Saved instances:{labels_tif_path}")


# ----------------- Example usage -----------------
if __name__ == "__main__":
    input_folder = '/home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/src/analysis'
    output_folder = '/home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/src/analysis/Test_images'
    model_path = '/home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/src/segmentation_model/saved_models/LoGSAGE_Multispec_sigma_Fusion3.pth'

    process_images_czi(
        input_folder=input_folder,
        output_folder=output_folder,
        model_path=model_path,
        prob_thresh=0.5,     # binarize model probability
        min_distance=9       # EXACT peak_local_max min_distance you requested
    )
