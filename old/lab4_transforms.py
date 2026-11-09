import PIL
from torchvision.transforms import v2 as t
import torch

imagenet_normalization = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

cifar10_normalization = {
    'mean':	(0.49139968, 0.48215827 ,0.44653124),
    'std':	(0.24703233, 0.24348505, 0.26158768)
    }

train_transform = t.Compose([
    t.RandomCrop(32, padding=4),
    t.RandomHorizontalFlip(),
    t.RandomRotation(15),
    t.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    t.ToImage(),
    t.ToDtype(torch.float32, scale=True),
    t.Normalize(**cifar10_normalization),
    t.RandomErasing(p=0.25),  # va DOPO ToTensor/Normalize
])
test_transform = t.Compose([
    t.ToImage(),
    t.ToDtype(torch.float32, scale=True),
    t.Normalize(**cifar10_normalization),
])

def revert_test_transform(x: torch.Tensor, reverse_transform=True) -> PIL.Image.Image:
    x = x.squeeze()
    mean = torch.tensor(cifar10_normalization['mean'], device=x.device)
    std = torch.tensor(cifar10_normalization['std'], device=x.device)

    x = x.as_subclass(torch.Tensor)
    if reverse_transform:
        x = x * std[:, None, None] + mean[:, None, None]

    return t.ToPILImage()(x.clamp(0, 1))