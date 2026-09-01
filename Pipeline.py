from dataclasses import dataclass
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
    def step(self): pass

class Pipeline:
    def __init__(
        self,
        model: Module,
        criterion: _Loss = None,
        optimizer: Optimizer = None,
        scheduler: LRScheduler = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler or DummyScheduler()
        self.checkpoint = None

    def load_checkpoint(self, checkpoint_file: str):
        print(f'Loading checkpoint "{checkpoint_file}"')
        self.checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(self.checkpoint['model'])
        if self.optimizer:
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(self.checkpoint['scheduler'])
        return self

    def save_checkpoint(self, checkpoint_file):
        print(f'Saving checkpoint at "{checkpoint_file}"')
        torch.save(self.checkpoint, checkpoint_file)
        return self

    def train_epoch(self, loader: DataLoader, progress_bar: bool=False):
        epoch_loss = 0.
        corrects = 0
        size = 0
        self.model.train()
        if progress_bar:
            loader = tqdm(loader)

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
    def eval_epoch(self, loader: DataLoader, progress_bar: bool=False):
        epoch_loss = 0.
        corrects = 0
        size = 0
        self.model.eval()

        if progress_bar:
            loader = tqdm(loader)

        for x, y in loader:
            size += x.shape[0]
            logits = self.model(x.cuda(non_blocking=True))
            y = y.cuda(non_blocking=True)
            loss = self.criterion(logits, y)
            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            corrects += (preds == y).sum().item()
        
        return epoch_loss / size, corrects / size

    @torch.inference_mode()
    def inference_logits(self, loader: DataLoader, progress_bar = False):
        logits = []
        self.model.eval()
        if progress_bar:
            loader = tqdm(loader)
            
        for x, _ in loader:
            logits.append(self.model(x.cuda(non_blocking=True)).cpu())
        return torch.cat(logits)
    
    def log(self, epoch, train_loss, train_acc, test_loss, test_acc):
        print(f'\n[{epoch}] L: {train_loss:.4f} | {test_loss:.4f} A: {train_acc*100:.2f} | {test_acc*100:.2f}')

    def get_epoch_iterator(self, epochs, progress_bar):
        epoch_iterator = range(epochs)
        if progress_bar:
            epoch_iterator = tqdm(epoch_iterator)
        return epoch_iterator

    def get_checkpoint(self):
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
    
    def train(self, trainloader: DataLoader, testloader: DataLoader, epochs: int, patience = float('inf'), progress_bar = False):
        epoch_iterator = self.get_epoch_iterator(epochs, progress_bar)
        self.checkpoint = None
        best_acc = 0
        max_patience = patience

        try:
            for e in epoch_iterator:
                train_loss, train_acc = self.train_epoch(trainloader)
                test_loss, test_acc = self.eval_epoch(testloader)
                self.scheduler.step()
                self.log(e, train_loss, train_acc, test_loss, test_acc)
                if test_acc > best_acc:
                    best_acc = test_acc
                    self.checkpoint = self.get_checkpoint()
                    patience = max_patience
                elif patience <= 1:
                    break
                else:
                    patience -= 1
        except KeyboardInterrupt:
            return self
        finally:
            return self
