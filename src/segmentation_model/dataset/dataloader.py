import os
import numpy as np
import tifffile
from torch.utils.data import Dataset
from torchvision import transforms
from skimage.filters import difference_of_gaussians
from skimage.exposure import rescale_intensity

class CarvanaDataset(Dataset):
    def __init__(self, root_path, test=False):
        self.root_path = root_path
        self.images = []
        self.masks = []

        images_path = os.path.join(root_path, "Test/images" if test else "augmented_images")
        masks_path = os.path.join(root_path, "Test/masks" if test else "augmented_masks")

        for img_filename in os.listdir(images_path):
            if img_filename.startswith('.'):
                continue
            mask_filename = img_filename.replace('.tiff', '_mask.tiff')
            img_path = os.path.join(images_path, img_filename)
            mask_path = os.path.join(masks_path, mask_filename)
            if os.path.exists(mask_path):
                self.images.append(img_path)
                self.masks.append(mask_path)

        self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        img_rgb = np.array(tifffile.imread(self.images[index]), dtype=np.float32) / 255.0
        mask = np.array(tifffile.imread(self.masks[index]), dtype=np.float32) / 255.0
        img_dapi = img_rgb[:, :, 0]
        dog_response = difference_of_gaussians(img_dapi, 9)
        dog_rescaled = rescale_intensity(dog_response, out_range=(0, 1))

        img_tensor = self.transform(img_rgb)
        dog_tensor = self.transform(dog_rescaled)
        mask_tensor = self.transform(mask)

        return img_tensor, dog_tensor, mask_tensor

    def __len__(self):
        return len(self.images)
