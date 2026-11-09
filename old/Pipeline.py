from dataclasses import dataclass
from typing import Literal

from matplotlib import pyplot as plt
import torch
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from torch import no_grad
from tqdm import tqdm
import wandb
from loggers import NoLogger

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
        scheduler: LRScheduler = None,
        logger: NoLogger = None
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler or DummyScheduler()
        self.logger = logger or NoLogger()
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

    def confusion_matrix(
        self,
        loader: DataLoader,
        img_path: str = None
    ):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        ytrue = []
        ypred = []

        for x, y in tqdm(loader):
            ytrue.append(y)
            with torch.no_grad():
                ypred.append(torch.softmax(self.model(x.cuda(non_blocking=True)), dim=1).argmax(dim=1).cpu())
        ytrue = torch.cat(ytrue)
        ypred = torch.cat(ypred)

        disp = ConfusionMatrixDisplay(confusion_matrix(ytrue, ypred))
        disp.plot()
        if img_path:
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        return self
    
    def train(self,
        trainloader: DataLoader,
        testloader: DataLoader,
        epochs: int,
        patience: int = None,
        progress_bar:bool = False
    ):
        epoch_iterator = self.get_epoch_iterator(epochs, progress_bar)
        self.checkpoint = None
        best_acc = 0
        patience = float('inf') or patience
        max_patience = patience
        
        try:
            for e in epoch_iterator:
                self.model.train()
                train_loss, train_acc = self.train_epoch(trainloader)
                self.scheduler.step()
                self.model.eval()
                test_loss, test_acc = self.eval_epoch(testloader)
                
                self.logger.log(e, {
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'valid_loss': test_loss,
                    'valid_acc': test_acc
                })
                
                if test_acc > best_acc:
                    best_acc = test_acc
                    self.checkpoint = self.get_checkpoint()
                    patience = max_patience
                elif patience <= 1:
                    break
                else:
                    patience -= 1
        finally:
            self.logger.end_run()
            return self