from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from tqdm import tqdm

from Training import Training
from lab4_datasets import get_cifar10, get_svhn
from model_pipelines import full_fine_tuning
from lab4_transforms import train_transform, test_transform
from torch.utils.data import DataLoader

def get_vision_model(num_classes):
    model = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.DEFAULT,
    )

    full_fine_tuning(model,output_classes=num_classes)
    model.requires_grad_(True)
    model.conv1.requires_grad_(False)

    return model

def msp_score(logits: torch.Tensor):
    return logits.softmax(dim=1).amax(dim=1)      # .amax = max().values

def energy_score(logits: torch.Tensor, temperature: float = 1):
    return -temperature * logits.exp().sum(dim=1).log()

def train_vision_model(save_model_path: str):
    torch.set_float32_matmul_precision('high')
    model = get_vision_model(num_classes=10).cuda()

    # training
    train_loader = torch.utils.data.DataLoader(
        dataset=get_cifar10(train=True, transform=train_transform),
        batch_size=128,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=get_cifar10(train=False, transform=test_transform),
        batch_size=1024,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-2,
        weight_decay=1e-3,
    )

    myloss = torch.nn.CrossEntropyLoss(reduction='sum')

    Training(
        model,
        optimizer,
        criterion=myloss,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100)
    )\
    .save_best_model(save_model_path)\
    .train(train_loader, test_loader, epochs=100, progress_bar=True, patience=5)
    return model

def dataloader_to_scores(loader: DataLoader, model: torch.nn.Module, ood_score) -> torch.Tensor:
    return torch.cat([ood_score(model, x.cuda()) for x, y in tqdm(loader)])

def show_scores(id_scores: np.array, ood_scores: np.array):
    plt.figure(figsize=(8, 5))

    plt.hist(id_scores, bins=50, alpha=0.5, density=True, label="ID")
    plt.hist(ood_scores, bins=50, alpha=0.5, density=True, label="OOD")

    plt.xlabel("Confidence score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    plt.show()

def get_logits(model, loader):
    return torch.cat([model(x.cuda()) for x, y in loader])

@torch.inference_mode()
def to_logits(load_model_path: str):
    model = get_vision_model(num_classes=10).cuda()
    model.load_state_dict(torch.load(load_model_path)['model'])
    model.eval()
    test_size = 10_000

    id_loader = DataLoader(
        dataset=get_cifar10(train=False, transform=test_transform),
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    torch.save(get_logits(model, id_loader).cpu(), 'id_logits.pt')

    ood_loader = DataLoader(
        dataset=get_svhn(size=test_size, transform=test_transform),
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    torch.save(get_logits(model, ood_loader).cpu(), 'ood_logits.pt')

def auroc_score(id_scores, ood_scores):
    id = id_scores.numpy()
    ood = ood_scores.numpy()
    ytrue = np.concat([np.ones_like(id), np.zeros_like(ood)])
    ypred = np.concat([id, ood])
    return sklearn.metrics.roc_auc_score(ytrue, ypred)

def aupr_score(id_scores, ood_scores):
    id = id_scores.numpy()
    ood = ood_scores.numpy()
    ytrue = np.concat([np.ones_like(id), np.zeros_like(ood)])
    ypred = np.concat([id, ood])
    return sklearn.metrics.average_precision_score(ytrue, ypred)

def logits_to_score(id_logits_path: str, ood_logits_path: str):
    id_logits = torch.load(id_logits_path)
    ood_logits = torch.load(ood_logits_path)
    print(auroc_score(id_logits, ood_logits),aupr_score(id_logits, ood_logits))

if __name__ == '__main__':
    logits_to_score('id_logits.pt', 'ood_logits.pt')
    