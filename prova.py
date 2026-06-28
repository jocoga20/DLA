import torch

mydict = torch.load('/home/jonathan/Desktop/DLA/checkpoints/resnet18.acc85.pt')

for k in ['epoch', 'test_accuracy']:
    print(f'{k} -> {mydict[k]}')