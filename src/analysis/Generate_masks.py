# --- CZI -> LoGSAGE-CBAM inference (same tiling/stitching as your UNet3D script) ---

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import czifile
from patchify import patchify
from skimage.filters import difference_of_gaussians

# ==== import your model pieces ====
from segmentation_model.models.swin_T_1 import SwinTransformer
from segmentation_model.models.Fusion import CBAMFusionBlock

# ----------------- Model (same as in your training) -----------------
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
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(out_channels * 2, out_channels)
    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv(x)
        return x

class SwinTransformerVit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SwinTransformerVit, self).__init__()
        self.swinViT = SwinTransformer()

        # saliency encoder
        self.encoder1 = DoubleConv(in_channels, 48)
        self.encoder2 = DoubleConv(48, 96)
        self.encoder3 = DoubleConv(96, 192)
        self.encoder4 = DoubleConv(192, 384)
        self.maxpool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.4)

        # image path
        self.conv_in_img = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.conv1_img = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.conv2_img = nn.Sequential(nn.Conv2d(96, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True))
        self.conv3_img = nn.Sequential(nn.Conv2d(192, 192, kernel_size=3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True))
        self.conv4_img = nn.Sequential(nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True))

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
        self.final_conv = nn.Conv2d(48, 1, kernel_size=1)

    def forward(self, x, saliency):
        skip0 = self.conv_in_img(x)

        s1, s2, s3, s4 = self.swinViT(x)
        s1 = self.conv1_img(s1); s2 = self.conv2_img(s2)
        s3 = self.conv3_img(s3); s4 = self.conv4_img(s4)

        xs = self.encoder1(saliency); xs = self.maxpool(xs); xs = self.dropout(xs); s1s = xs
        xs = self.encoder2(xs);      xs = self.maxpool(xs); xs = self.dropout(xs); s2s = xs
        xs = self.encoder3(xs);      xs = self.maxpool(xs); xs = self.dropout(xs); s3s = xs
        xs = self.encoder4(xs);      xs = self.maxpool(xs); xs = self.dropout(xs); s4s = xs

        f1 = self.fuse1(s1, s1s)
        f2 = self.fuse2(s2, s2s)
        f3 = self.fuse3(s3, s3s)
        f4 = self.fuse4(s4, s4s)

        x = self.decoder1(f4, f3)
        x = self.decoder2(x, f2)
        x = self.decoder3(x, f1)
        x = self.decoder4(x, skip0)
        out = self.final_conv(x)
        return out
# -------------------------------------------------------------------

def load_model(model_path, device):
    # weights_only may not exist in older torch; try it, else fallback
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

def make_dog_single_channel(patch_uint8):
    ch0 = patch_uint8[..., 0].astype(np.float32) / 255.0
    dog = difference_of_gaussians(ch0, 8)
    mn, mx = dog.min(), dog.max()
    dog = (dog - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(dog, dtype=np.float32)
    return dog[..., None].astype(np.float32)  # (H,W,1)

def make_dog_from_first_channel(patch_uint8):
    ch0 = patch_uint8[..., 0].astype(np.float32) / 255.0
    dog = difference_of_gaussians(ch0, 8)
    mn, mx = dog.min(), dog.max()
    dog = (dog - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(dog, dtype=np.float32)
    return np.repeat(dog[..., None], 3, axis=-1).astype(np.float32)  # (H,W,3)

def predict_and_reassemble_czi(patches, model, device, sal_ch, thresh=0.5):
    """
    patches: result of patchify(imagein, (256,256,3), step=(128,128,3))
             shape: (nrows, ncols, 1, 256,256,3), dtype uint8
    """
    nrows, ncols = patches.shape[0], patches.shape[1]
    core = 128
    cores = np.zeros((nrows, ncols, core, core), dtype=np.uint8)

    with torch.no_grad():
        for i in range(nrows):
            for j in range(ncols):
                patch = patches[i, j, 0]  # (256,256,3) uint8
                img_np = patch.astype(np.float32) / 255.0  # (256,256,3)

                if sal_ch == 1:
                    dog_np = make_dog_single_channel(patch)  # (256,256,1)
                    t_dog = torch.from_numpy(dog_np.transpose(2,0,1)).unsqueeze(0).float().to(device)
                else:
                    dog_np = make_dog_from_first_channel(patch)  # (256,256,3)
                    t_dog = torch.from_numpy(dog_np.transpose(2,0,1)).unsqueeze(0).float().to(device)

                t_img = torch.from_numpy(img_np.transpose(2,0,1)).unsqueeze(0).float().to(device)

                logits = model(t_img, t_dog)
                prob = torch.sigmoid(logits)[0,0].cpu().numpy()
                pred = (prob > thresh).astype(np.uint8)

                # center 128×128
                off = (256 - core)//2
                core_patch = pred[off:off+core, off:off+core]
                cores[i, j] = core_patch

    # stitch cores row/col -> final
    H = nrows * core
    W = ncols * core
    out = np.zeros((H, W), dtype=np.uint8)
    for i in range(nrows):
        for j in range(ncols):
            y0, x0 = i * core, j * core
            out[y0:y0+core, x0:x0+core] = cores[i, j]
    return out  # 0/1

def process_images_czi(input_folder, output_folder, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, sal_ch = load_model(model_path, device)

    images_folder = os.path.join(input_folder, 'Test_images')
    masks_folder  = os.path.join(output_folder, 'LoGSAGE-CBAM_masks')  # or 'UNet_mask' if you want identical naming
    os.makedirs(masks_folder, exist_ok=True)

    for filename in os.listdir(images_folder):
        if not filename.lower().endswith('.czi'):
            continue

        file_path = os.path.join(images_folder, filename)
        print(f"Processing {filename}")

        # --- read CZI and build 3ch like your old code ---
        imageorg = czifile.imread(file_path)  # expects (C, Y, X, 1)
        I11 = imageorg[0, :, :, 0]
        I21 = imageorg[1, :, :, 0]
        I31 = imageorg[2, :, :, 0]

        I11_norm = cv2.normalize(I11, None, 0, 255, cv2.NORM_MINMAX)
        I21_norm = cv2.normalize(I21, None, 0, 255, cv2.NORM_MINMAX)
        I31_norm = cv2.normalize(I31, None, 0, 255, cv2.NORM_MINMAX)
        rgb_image1 = np.stack((I11_norm, I21_norm, I31_norm), axis=2).astype(np.uint8)  # (H,W,3) uint8

        # --- same padding as before ---
        imagein = np.pad(rgb_image1, ((64,128), (64,64), (0,0)), mode='constant')

        # --- same tiling as before ---
        patches = patchify(imagein, (256, 256, 3), step=(128, 128, 3))  # (nrows, ncols, 1, 256,256,3)

        # --- predict + stitch (center 128) ---
        mask_bin = predict_and_reassemble_czi(patches, model, device, sal_ch=sal_ch, thresh=0.5)  # 0/1

        # --- save ---
        mask_filename = filename.replace('.czi', '_mask.png')
        mask_path = os.path.join(masks_folder, mask_filename)
        cv2.imwrite(mask_path, (mask_bin * 255).astype(np.uint8))
        print(f"Mask saved to {mask_path}")

# ----------------- Example usage -----------------
if __name__ == "__main__":
    input_folder = '/home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/src/analysis'
    output_folder = '/home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/src/analysis/Test_images'
    model_path = '/home/sahar/Cellular_Profiling_and_Subtyping_of_Brest_cancer_TME/src/segmentation_model/saved_models/LoGSAGE_Multispec_sigma_Fusion3.pth'
    process_images_czi(input_folder, output_folder, model_path)