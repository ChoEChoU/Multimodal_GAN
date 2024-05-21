import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomMedicalDataset(Dataset):
    def __init__(self, ct_data, clinical_data, labels, transform=None):
        self.ct_data = ct_data
        self.clinical_data = clinical_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.ct_data)

    def __getitem__(self, idx):
        ct_sample = self.ct_data[idx]
        clinical_sample = self.clinical_data[idx]
        label = self.labels[idx]
        if self.transform:
            ct_sample = self.transform(ct_sample)
        return ct_sample, clinical_sample, label

def get_data_loader(batch_size):
    # Dummy data generation
    img_shape = (1, 64, 64, 64)
    num_classes = 10

    ct_data = np.random.randn(1000, *img_shape).astype(np.float32)  # Dummy CT data
    clinical_data = np.random.randn(1000, 1).astype(np.float32)  # Dummy clinical data
    labels = np.random.randint(0, num_classes, 1000)  # Dummy labels

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = CustomMedicalDataset(ct_data, clinical_data, labels, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader