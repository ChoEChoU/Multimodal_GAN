import os
import pydicom
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def preprocess_ct_images(ct_slices, target_depth):
    num_slices = ct_slices.shape[0]
    if num_slices < target_depth:
        padding = (target_depth - num_slices) // 2
        ct_slices = np.pad(ct_slices, ((padding, padding), (0, 0), (0, 0)), 'constant')
    elif num_slices > target_depth:
        start = (num_slices - target_depth) // 2
        ct_slices = ct_slices[start:start + target_depth]
    return ct_slices

class NSCLCDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_depth=257):
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_depth = target_depth
        self.clinical_features = self.annotations.columns[1:-2]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        dir_path = self.annotations.iloc[idx]['File Location']
        img_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.dcm')]

        images = []
        for img_file in sorted(img_files):
            img = pydicom.dcmread(img_file).pixel_array
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images).squeeze()
        images = preprocess_ct_images(images, self.target_depth)
        images = images.unsqueeze(0).type(torch.float32)

        clinical_data = self.annotations.iloc[idx][self.clinical_features].values.astype(np.float32)
        clinical_data = torch.tensor(clinical_data)

        label = torch.tensor(self.annotations.iloc[idx]['Survival Status'], dtype=torch.long)
        return images, clinical_data, label

def get_data_loader(batch_size, annotations_file):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = NSCLCDataset(annotations_file=annotations_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader