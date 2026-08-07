import torchvision
import torch

def get_cifar10(train, transform):
    return torchvision.datasets.CIFAR10('./mydatasets', train=train, transform=transform, download=True)

def get_fakedata(transform, size):
        return torchvision.datasets.FakeData(size = size, image_size=(3, 32, 32), transform=transform, num_classes=1)

def get_svhn(transform, size : str | int = 'full', split='test'):
    svhn = torchvision.datasets.SVHN(root='./mydatasets', split=split, download=True, transform=transform)
    if size == 'full':
        return svhn
    
    indices = torch.randperm(len(svhn))[:size]
    return torch.utils.data.Subset(svhn, indices)
    