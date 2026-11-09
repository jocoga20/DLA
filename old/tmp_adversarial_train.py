import torch
import sys
from adversarial import adversarial_training
from lab4_transforms import train_transform, test_transform
from lab4_models import get_vision_model
from lab4_datasets import get_cifar10
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from reproducibility import set_seed

def main():
    g=set_seed(0)
    m = get_vision_model(10)
    adversarial_training(
        model=m,
        trainset=get_cifar10(split='train', transform=train_transform),
        validset=get_cifar10(split='valid', transform=test_transform),
        criterion=CrossEntropyLoss(reduction='sum'),
        optimizer=SGD(params=m.parameters(), lr=1e-2, momentum=.5, nesterov=True, weight_decay=1e-3),
        epochs=int(sys.argv[1]),
        adv_ratio=.5,
        epsilon=3/255,
        generator=g,
        progress_bar=True
    )
    save_path = 'adv_model.pt'
    print(f"Saving model dict @ '{save_path}'")
    torch.save(m.state_dict(), save_path)

if __name__ == '__main__':
    main()
