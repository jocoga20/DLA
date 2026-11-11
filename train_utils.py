from typing import Literal

import torch
from torch.nn import CrossEntropyLoss, Module
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score
import numpy as np
from loggers import NoLogger

class DummyScheduler(LRScheduler):
    def __init__(self): pass
    def step(self): pass
    def load_state_dict(self, state_dict): pass
    def state_dict(self): return dict({})


def train_epoch(
    model: Module,
    loader: DataLoader,
    optimizer: Optimizer,
    loss_rule: _Loss = None,
    progress_bar: bool = False
) -> tuple[float, float, float]:
    total_loss = 0.
    corrects = 0
    size = 0
    loss_rule = loss_rule or CrossEntropyLoss(reduction='sum')

    if progress_bar:
        loader = tqdm(loader)

    all_preds = []
    all_targets = []

    model.train()

    for x, y in loader:
        size += x.shape[0]
        optimizer.zero_grad()
        logits = model(x.cuda(non_blocking=True))
        y = y.cuda(non_blocking=True)
        loss = loss_rule(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        corrects += (preds == y).sum().item()

        # Raccolta su CPU per il calcolo finale con sklearn
        all_preds.append(preds.detach().cpu())
        all_targets.append(y.detach().cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return total_loss / size, corrects / size, balanced_acc

@torch.no_grad()
def eval_epoch(
    model: Module,
    loader: DataLoader,
    loss_rule: _Loss = None,
    progress_bar: bool = False
) -> tuple[float, float, float]:
    total_loss = 0.
    corrects = 0
    size = 0
    loss_rule = loss_rule or CrossEntropyLoss(reduction='sum')

    if progress_bar:
        loader = tqdm(loader)

    all_preds = []
    all_targets = []

    model.eval()

    for x, y in loader:
        size += x.shape[0]
        logits = model(x.cuda(non_blocking=True))
        y = y.cuda(non_blocking=True)
        loss = loss_rule(logits, y)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        corrects += (preds == y).sum().item()

        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return total_loss / size, corrects / size, balanced_acc

def train(
    model: Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    epochs: int,
    max_patience: int,
    scheduler: LRScheduler = None,
    loss_rule: _Loss = None,
    logger: NoLogger = None,
    progress_bars: Literal['none', 'epoch', 'train_and_eval', 'train', 'eval'] = 'none'
):
    loss_rule = loss_rule or CrossEntropyLoss(reduction='sum')
    scheduler = scheduler or DummyScheduler()
    logger = logger or NoLogger()

    epoch_iterator = range(epochs)
    if progress_bars == 'epoch':
        epoch_iterator = tqdm(epoch_iterator)

    pb_train = progress_bars == 'train' or progress_bars == 'train_and_eval'
    pb_eval = progress_bars == 'eval' or progress_bars == 'train_and_eval'

    best_loss = float('inf')
    best_acc = -1

    with logger:
        for e in epoch_iterator:
            train_loss, train_acc, train_bal_acc = train_epoch(model, train_loader, optimizer, loss_rule, progress_bar=pb_train)
            eval_loss, eval_acc, eval_bal_acc = eval_epoch(model, eval_loader, loss_rule, progress_bar=pb_eval)
            logger.log(step=e, data_dict={
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'train_acc': train_acc,
                'eval_acc': eval_acc,
                'train_bacc': train_bal_acc,
                'eval_bacc': eval_bal_acc
            })
            scheduler.step()

            if eval_acc > best_acc:
                best_acc = eval_acc

            if eval_loss < best_loss:
                patience = 0
            else:
                patience += 1
                if max_patience is not None and patience >= max_patience:
                    break
    return best_acc