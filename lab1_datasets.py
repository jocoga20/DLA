import numpy as np
import torch
from torchvision.datasets import GTSRB
from torchvision.transforms.v2 import Transform
from torch.utils.data import Subset, random_split

from reproducibility import set_seed

def get_gtsrb(split: str, transform: Transform = None):
    return GTSRB(root='./mydatasets', split=split, download=True, transform=transform)

def get_gtsrb_full_train(transform: Transform = None):
    return get_gtsrb('train', transform)

def get_gtsrb_test(transform: Transform = None):
    return get_gtsrb('test', transform)

def get_split_indices(train_ratio: float, full_size: int):
    split_idx = int(train_ratio * full_size)
    p = np.random.permutation(full_size)
    return p[:split_idx], p[split_idx:]

def get_gtsrb_train_valid_split(split_seed: int, train_ratio: float, train_transform: Transform, valid_transform):
    g = set_seed(split_seed)
    train_size = 26640
    test_size = 12630
    train_idx, valid_idx = get_split_indices(train_ratio, train_size)
    train_ds = Subset(get_gtsrb_full_train(train_transform), train_idx)
    valid_ds = Subset(get_gtsrb_full_train(valid_transform), valid_idx)
    return train_ds, valid_ds

def extract_labels(dataset, save_path):
    import torch
    torch.save(torch.tensor([y for x, y in dataset]), save_path)

def shannon_entropy(x: np.array, norm = False):
    h = - (x * np.log2(x)).sum()
    if norm:
        return h / np.log2(43)
    else:
        return h

def gini_index(x: np.array):
    return 1 - np.sum(x ** 2)

def se(ds, n = False):
    labels = np.bincount(extract_labels(ds)).astype(np.float32)
    labels /= labels.sum()
    print(labels)
    return shannon_entropy(labels, norm=n)

def rel_freq(ds):
    x = np.bincount(extract_labels(ds)).astype(float)
    return x / x.sum()