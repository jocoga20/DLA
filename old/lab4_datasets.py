from typing import Literal
import numpy as np
import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Subset

class_names = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])

def cifar10_original_split(train: bool, transform=None):
	return CIFAR10(root='./mydatasets', train=train, transform=transform, download=True)

def cifar10_index_file(train: bool, index_file: str, transform=None):
	return Subset(cifar10_original_split(train=train, transform=transform), torch.load(index_file).long().tolist())

def get_cifar10(split: Literal['train', 'valid', 'test', 'trainval'], transform=None):
	if split == 'trainval':
		return cifar10_original_split(train=True, transform=transform)
	elif split == 'test':
		return cifar10_original_split(train=False, transform=transform)
	elif split == 'train':
		return cifar10_index_file(train=True, transform=transform, index_file='lab4_res/train_indices.pt')
	else:
		return cifar10_index_file(train=True, transform=transform, index_file='lab4_res/valid_indices.pt')

def get_cifar100_ood(transform=None):
	return Subset(CIFAR100(root='./mydatasets', train=True, transform=transform, download=True), torch.load('lab4_res/ood_indices.pt').long().tolist())