import torch

from adversarial import get_gradient, targeted_fgsm
from lab4_datasets import get_cifar10
from lab4_models import load_trained_adv_model, load_trained_model
from lab4_transforms import test_transform, revert_test_transform
from torch.nn import functional as f

ds = get_cifar10('trainval', test_transform)
classes = ds.classes
x, y = ds[0]

model = load_trained_adv_model().cpu()
model.eval()
x = x.unsqueeze(0)

g = get_gradient(x, torch.tensor(3).unsqueeze(0), model)
eps = 3 / 255
x1 = x - eps * g.sign()
x = torch.cat([x, x1])
softmaxed_logits = model(x).softmax(dim=1)
print(softmaxed_logits.max(dim=1))
#print(logits.max(dim=1))

if True:
    revert_test_transform(x[1]).save('x1.png')
    revert_test_transform(x[0]).save('x0.png')
    revert_test_transform(x[1] - x[0]).save('xdiff.png')