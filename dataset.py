import torch
import pandas as pd
import numpy as np
import nibabel as nib
import cv2
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, csv_file, num_slices=32, image_size=128):
        self.data = pd.read_csv(csv_file)
        self.num_slices = num_slices
        self.image_size = image_size

    def __len__(self):
        return len(self.data)

    def normalize(self, volume):
        volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
        volume = np.clip(volume, -3, 3)
        return volume

    def select_slices(self, volume):
        total_slices = volume.shape[2]
        indices = np.linspace(10, total_slices - 10, self.num_slices).astype(int)
        slices = volume[:, :, indices]
        return slices

    def __getitem__(self, idx):
        path = self.data.iloc[idx]["path"]
        label = int(self.data.iloc[idx]["label"])

        volume = nib.load(path).get_fdata()
        volume = self.normalize(volume)
        slices = self.select_slices(volume)

        processed = []

        for i in range(slices.shape[2]):
            img = slices[:, :, i]
            img = cv2.resize(img, (self.image_size, self.image_size))
            processed.append(img)

        processed = np.stack(processed, axis=0)
        processed = processed[:, None, :, :]

        return torch.tensor(processed, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
