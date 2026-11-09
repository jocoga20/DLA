from torch.optim import Optimizer
import torch
from torch.utils.data import DataLoader, Dataset
from torch import Generator, Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from tqdm import tqdm

from Pipeline import Pipeline
from loggers import WandbLogger

def get_gradient(x: Tensor, y: Tensor, model: Module, criterion = torch.nn.CrossEntropyLoss(reduction='sum')):
    x = x.detach().clone().requires_grad_()
    logits = model(x)
    loss = criterion(logits, y)
    return torch.autograd.grad(loss, x)[0]

def fgsm(x: Tensor, y_gt: Tensor, model: Module, epsilon: float, criterion = torch.nn.CrossEntropyLoss(reduction='sum')):
    g = get_gradient(x, y_gt, model, criterion)
    return (x + epsilon * g.sign()).detach()

def targeted_fgsm(x: Tensor, y_target: Tensor, model: Module, epsilon: float, criterion = torch.nn.CrossEntropyLoss(reduction='sum')):
    g = get_gradient(x, y_target, model, criterion)
    return (x - epsilon * g.sign()).detach()

def train_epoch_adv(model: Module, loader: DataLoader, optimizer: Optimizer, criterion: _Loss, adv_ratio: float = 0.5, epsilon = 1/255, progress_bar = False):
    total_samples = 0
    corrects = 0
    total_loss = 0
    if progress_bar:
        loader = tqdm(loader)

    for x, y in loader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        batch_size = len(y)
        total_samples += batch_size
        n_adv = int(batch_size * adv_ratio)
        idxs_adv = torch.randperm(batch_size)[:n_adv]

        x_train = x.clone()
        x_train[idxs_adv] = fgsm(x[idxs_adv], y[idxs_adv], model, epsilon)
        
        optimizer.zero_grad()
        logits = model(x_train)

        loss = criterion(logits, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        preds = logits.argmax(dim=1)
        corrects += (preds == y).sum().item()

    return total_loss / total_samples, corrects / total_samples

def eval_epoch_adv(model: Module, loader: DataLoader, criterion: _Loss, adv_ratio:float = 0.5, epsilon = 1/255, progress_bar = False):
    total_samples = 0
    corrects = 0
    total_loss = 0
    if progress_bar:
        loader = tqdm(loader)

    for x, y in loader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        batch_size = len(y)
        total_samples += batch_size
        n_adv = int(batch_size * adv_ratio)
        idxs_adv = torch.randperm(batch_size)[:n_adv]

        x_train = x.clone()
        x_train[idxs_adv] = fgsm(x[idxs_adv], y[idxs_adv], model, epsilon)
        
        logits = model(x_train)

        loss = criterion(logits, y)
        total_loss += loss.item()
        loss.backward()
        preds = logits.argmax(dim=1)
        corrects += (preds == y).sum().item()

    return total_loss / total_samples, corrects / total_samples

def get_loaders(trainset: Dataset, validset: Dataset, g: Generator):
    bs = 256
    total_workser = 12
    train_workers = 8

    trainloader = DataLoader(
        dataset=trainset,
        batch_size=bs,
        shuffle=True,
        num_workers=total_workser,
        pin_memory=True,
        persistent_workers=True,
        generator=g
    )

    validloader = DataLoader(
        dataset=validset,
        batch_size=bs,
        shuffle=False,
        num_workers=total_workser - train_workers,
        pin_memory=True,
        persistent_workers=True
    )

    return trainloader, validloader

def adversarial_training(
    model: Module,
    trainset: Dataset,
    validset: Dataset,
    criterion: _Loss,
    optimizer: Optimizer,
    epochs: int,
    generator: Generator,
    adv_ratio: float = 0.5,
    epsilon: float = 1/255,
    progress_bar = False
):
    logger = WandbLogger(project='DLA', run_name='lab4 adv train')
    trainloader, validloader = get_loaders(trainset, validset, generator)
    p = Pipeline(model, criterion, optimizer)

    iterator = tqdm(range(epochs)) if progress_bar else range(epochs)
    for e in iterator:
        model.train()
        train_loss, train_acc = train_epoch_adv(model, trainloader, optimizer, criterion, adv_ratio, epsilon)
        model.eval()
        # REMEMBER: don't use torch.no_grad()
        valid_loss, valid_acc = p.eval_epoch(validloader)
        valid_adv_loss, valid_adv_acc = eval_epoch_adv(model, validloader, criterion, adv_ratio=1, epsilon=epsilon)
        logger.log(step=e, data_dict={
            'train_loss': train_loss,
            'train_acc': train_acc,
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
            'valid_adv_loss': valid_adv_loss,
            'valid_adv_acc': valid_adv_acc
        })
    logger.end_run()