import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile

class Dataset(Dataset):
    def __init__(self, root_path, test=False):
        images_path = os.path.join(root_path, "Test/images/" if test else "augmented_images/")
        masks_path = os.path.join(root_path, "Test/masks/" if test else "augmented_masks/")

        self.images = sorted([os.path.join(images_path, i) for i in os.listdir(images_path)])
        self.masks = sorted([os.path.join(masks_path, i) for i in os.listdir(masks_path)])

    def __getitem__(self, index):
        img = np.array(tifffile.imread(self.images[index]), dtype=np.float32) / 255
        mask = np.array(tifffile.imread(self.masks[index]), dtype=np.float32) / 255

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img.unsqueeze(0), mask.unsqueeze(0)

    def __len__(self):
        return len(self.images)
