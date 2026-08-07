from torchvision.transforms import v2 as t
import torch

imagenet_normalization = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

train_transform = t.Compose([
    t.RandomCrop(32, padding=4),
    t.RandomHorizontalFlip(p=0.5),
    t.ToImage(),
    t.ToDtype(torch.float32, scale=True),
    t.Normalize(**imagenet_normalization),
])

test_transform = t.Compose([
    t.ToImage(),
    t.ToDtype(torch.float32, scale=True),
    t.Normalize(**imagenet_normalization),
])

default_transform = t.Compose([t.ToImage(), t.ToDtype(torch.float32, scale=True)])