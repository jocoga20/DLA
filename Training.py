from typing import Literal

import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch import no_grad
from tqdm import tqdm

class DummyScheduler(LRScheduler):
    def __init__(self): pass
    def state_dict(self): pass
    def load_state_dict(self, state_dict): pass

class Training:
    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler = DummyScheduler(),
        criterion: _Loss = CrossEntropyLoss(reduction='sum')
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.checkpoint_file = None

    def save_best_model(self, checkpoint_file):
        self.checkpoint_file = checkpoint_file
        return self

    def train_epoch(self, loader: DataLoader):
        epoch_loss = 0.
        corrects = 0
        size = 0
        self.model.train()

        for x, y in loader:
            size += x.shape[0]
            self.optimizer.zero_grad()
            logits = self.model(x.cuda(non_blocking=True))
            y = y.cuda(non_blocking=True)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            corrects += (preds == y).sum().item()
        
        return epoch_loss / size, corrects / size
    
    @no_grad()
    def eval_epoch(self, loader: DataLoader):
        epoch_loss = 0.
        corrects = 0
        size = 0
        self.model.train()

        for x, y in loader:
            size += x.shape[0]
            logits = self.model(x.cuda(non_blocking=True))
            y = y.cuda(non_blocking=True)
            loss = self.criterion(logits, y)
            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            corrects += (preds == y).sum().item()
        
        return epoch_loss / size, corrects / size
    
    def log(self, epoch, train_loss, train_acc, test_loss, test_acc):
        print(f'[{epoch}] L: {train_loss:.4f} | {test_loss:.4f} A: {train_acc*100:.2f} | {test_acc*100:.2f}')

    def get_epoch_iterator(self, epochs, progress_bar):
        epoch_iterator = range(epochs)
        if progress_bar:
            epoch_iterator = tqdm(epoch_iterator)
        return epoch_iterator
    
    def train(self, trainloader: DataLoader, testloader: DataLoader, epochs: int, progress_bar = False):
        epoch_iterator = self.get_epoch_iterator(epochs, progress_bar)
        checkpoint = None
        best_acc = 0

        try:
            for e in epoch_iterator:
                train_loss, train_acc = self.train_epoch(trainloader)
                test_loss, test_acc = self.eval_epoch(testloader)
                self.scheduler.step()
                if best_acc > test_acc:
                    best_acc = test_acc
                    checkpoint = {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict()
                    }

                self.log(e, train_loss, train_acc, test_loss, test_acc)
        finally:
            if self.checkpoint_file:
                torch.save(checkpoint, self.checkpoint_file)
            

    
class AdversarialTraining(Training):
    def __init__(self, model, optimizer, scheduler = DummyScheduler(), criterion = CrossEntropyLoss(reduction='sum')):
        super().__init__(model, optimizer, scheduler, criterion)
    
    def train_epoch(self, loader: DataLoader):
        epoch_loss = 0.
        corrects = 0
        size = 0
        self.model.train()

        for x, y in loader:
            size += x.shape[0]
            self.optimizer.zero_grad()
            logits = self.model(x.cuda(non_blocking=True))
            y = y.cuda(non_blocking=True)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            corrects += (preds == y).sum().item()


def adv_train(model, loader, optimizer, criterion, adv_ratio = 0.5, epsilon = 4/255):
    epoch_loss = 0.
    corrects = 0
    size = 0

    model.train()

    for x, y in loader:
        x = x.cuda()
        y = y.cuda()
        batch_size = x.shape[0]
        adv_idx = torch.randperm(batch_size, device=x.device)[:int(adv_ratio * batch_size)]
        x_adv = x[adv_idx].detach().clone().requires_grad_(True)
        y_adv = y[adv_idx]
        logits = model(x_adv)
        loss = criterion(logits, y_adv)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = (x_adv + epsilon * grad.sign()).detach()
        x[adv_idx] = x_adv
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        corrects += (logits.argmax(1) == y).sum().item()
        size += batch_size

    return epoch_loss / size, corrects / size