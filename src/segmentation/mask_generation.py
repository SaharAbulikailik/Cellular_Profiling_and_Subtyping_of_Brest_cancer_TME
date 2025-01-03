import os
import torch
import numpy as np
import cv2
import czifile
from patchify import patchify
from unet3d import UNet3D  # Ensure this is defined or imported in the same file

# Load the UNet3D model
def load_model(model_path, device):
    model = UNet3D(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict masks for all patches
def predict_and_reassemble(patches, model, device):
    predicted_patches = []
    for row in range(patches.shape[0]):
        row_patches = []
        for col in range(patches.shape[1]):
            patch = patches[row, col, 0, :, :, :] / 255
            patch = patch.astype(np.float32)
            patch_tensor = torch.from_numpy(patch).float().permute(2, 0, 1).to(device)

            with torch.no_grad():
                prediction = model(patch_tensor.unsqueeze(0).unsqueeze(0))
                prediction = torch.sigmoid(prediction)
                prediction = (prediction > 0.5).float().cpu().numpy()

            row_patches.append(prediction[0, 0, 0, :, :])
        predicted_patches.append(row_patches)

    return np.array(predicted_patches)

# Build the full image from patches
def build_image(patches_img, full_predicted_mask):
    desired_shape = patches_img.shape[:-1]
    labels = np.array(full_predicted_mask)
    reshaped_labels = np.reshape(labels, desired_shape)
    crop_size = 128
    cropped_labels = np.empty((8, 10, 1, crop_size, crop_size))
    for i in range(8):
        for j in range(10):
            patch = reshaped_labels[i, j, 0]
            height, width = patch.shape
            crop_start_h = max(0, (height - crop_size) // 2)
            crop_end_h = crop_start_h + crop_size
            crop_start_w = max(0, (width - crop_size) // 2)
            crop_end_w = crop_start_w + crop_size
            cropped_patch = patch[crop_start_h:crop_end_h, crop_start_w:crop_end_w]
            cropped_labels[i, j, 0] = cropped_patch

    cropped_labels = np.squeeze(cropped_labels)
    cropped_labels = np.transpose(cropped_labels, (0, 2, 1, 3))
    return np.reshape(cropped_labels, (8 * 128, 10 * 128))

# Main function to process images
def process_images(input_folder, output_folder, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    images_folder = os.path.join(input_folder, 'A1819')
    masks_folder = os.path.join(output_folder, 'UNet_mask')
    os.makedirs(masks_folder, exist_ok=True)

    for filename in os.listdir(images_folder):
        file_path = os.path.join(images_folder, filename)
        if filename.lower().endswith('.czi'):
            print(f"Processing {filename}")
            imageorg = czifile.imread(file_path)
            I11 = cv2.normalize(imageorg[0, :, :, 0], None, 0, 255, cv2.NORM_MINMAX)
            I21 = cv2.normalize(imageorg[1, :, :, 0], None, 0, 255, cv2.NORM_MINMAX)
            I31 = cv2.normalize(imageorg[2, :, :, 0], None, 0, 255, cv2.NORM_MINMAX)
            rgb_image1 = np.stack((I11, I21, I31), axis=2)
            imagein = np.pad(rgb_image1, ((64, 128), (64, 64), (0, 0)), mode='constant')
            patches = patchify(imagein, (256, 256, 3), step=(128, 128, 3))

            full_mask = predict_and_reassemble(patches, model, device)
            output = build_image(patches, full_mask)

            mask_filename = filename.replace('.czi', '_mask.png')
            mask_path = os.path.join(masks_folder, mask_filename)
            cv2.imwrite(mask_path, output * 255)
            print(f"Mask saved to {mask_path}")

# Example usage
if __name__ == "__main__":
    input_folder = '/nfs/cc-filer/home/sabulikailik/new_images'
    output_folder = '/nfs/cc-filer/home/sabulikailik/new_images/A1819'
    model_path = '/nfs/cc-filer/home/sabulikailik/UNet_project/saved_models/unet3D_CL2.pth'
    process_images(input_folder, output_folder, model_path)
