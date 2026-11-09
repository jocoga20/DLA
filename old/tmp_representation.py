from matplotlib import pyplot as plt
import torch

from lab4_datasets import get_cifar10
from lab4_models import load_trained_model
from lab4_transforms import test_transform, revert_test_transform, cifar10_normalization

m = load_trained_model()
ds = get_cifar10('train', test_transform)
x1, y1 = ds[0]

xs = [x1]

for ix in [0, 16]:
    for iy in [0, 16]:
        x2 = x1.clone()
        x2[:,ix:ix+16,iy:iy+16] = 0
        xs.append(x2)

x2 = x1.clone()
x2[:,8:24,8:24] = 0
xs.append(x2)
logits = m(torch.stack(xs))
print(logits.softmax(dim=1).max(dim=1))

for xi in xs:
    revert_test_transform(xi, reverse_transform=False).show()