import torch
import torchvision
import torchvision.transforms.v2 as t

from system_utils import get_max_workers

def get_dataset(split, transform=None):
    return torchvision.datasets.GTSRB(root='mydatasets', split=split, transform=transform, download=True)

def get_train(transform=None):
    return get_dataset('train', transform=transform)

def get_test(transform=None):
    return get_dataset('test', transform=transform)

gtsrb_mean = [0.3805, 0.3484, 0.3574]
gtsrb_std = [0.3031, 0.2950, 0.3007]

gtsrb_train_size = 26640
gtsrb_test_size = 12630

def default_loaders(train_batch_size, test_batch_size):
    persistent_workers = True
    img_size = 32
    train_transform = t.Compose([
        t.Resize((img_size, img_size)),              # oppure 48x48 o 64x64
        t.RandomAffine(
            degrees=15,
            translate=(0.10, 0.10),
            scale=(0.9, 1.1),
            shear=10
        ),
        t.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        t.ToImage(),
        t.ToDtype(torch.float32, scale=True),
        t.Normalize(gtsrb_mean, gtsrb_std)
    ])

    test_transform = t.Compose([
        t.Resize((img_size, img_size)),
        t.ToImage(),
        t.ToDtype(torch.float32, scale=True),
        t.Normalize(gtsrb_mean, gtsrb_std)
    ])

    workers = get_max_workers() // 2
    return (
        torch.utils.data.DataLoader(get_train(train_transform),
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            timeout=20
        ),
        torch.utils.data.DataLoader(get_test(test_transform),
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=persistent_workers,
            timeout=20
        )
    )