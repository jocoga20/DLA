import torch
from torch.utils.data import Dataset

def compute_normalization(dataset: Dataset):
    images = torch.stack([img for img, _ in dataset]).cuda()
    print(images.mean(dim=(0,2,3)), images.std(dim=(0,2,3)))
