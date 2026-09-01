from torch.optim import Optimizer
import torch
from torch.utils.data import DataLoader
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as f
from torch.nn.modules.loss import _Loss
from tqdm import tqdm
from Pipeline import Pipeline
from lab4 import get_vision_model, set_seed
from lab4_datasets import get_cifar10
from lab4_transforms import train_transform, test_transform

def get_gradient(x: Tensor, y: Tensor, model: Module, criterion = f.cross_entropy):
    x = x.detach().clone().requires_grad_()
    logits = model(x)
    loss = criterion(logits, y)
    return torch.autograd.grad(loss, x)[0]

def fgsm(x: Tensor, y_gt: Tensor, model: Module, epsilon: float, criterion = f.cross_entropy):
    g = get_gradient(x, y_gt, model, criterion)
    return (x + epsilon * g.sign()).detach()

def targeted_fgsm(x: Tensor, y_target: Tensor, model: Module, epsilon: float, criterion = f.cross_entropy):
    g = get_gradient(x, y_target, model, criterion)
    return (x - epsilon * g.sign()).detach()    

def train_epoch_adv(model: Module, loader: DataLoader, optimizer: Optimizer, criterion: _Loss, adv_ratio = 0.5, epsilon = 4/256, progress_bar = False):
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

def test_epoch_adv(model: Module, loader: DataLoader, criterion: _Loss, adv_ratio = 0.5, epsilon = 1/256, progress_bar = False):
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

def main():
    save_model_path = 'checkpoints/prova.pt'
    seed = 42
    set_seed(seed)
    torch.set_float32_matmul_precision('high')
    model = get_vision_model(num_classes=10).cuda()
    generator = torch.Generator()
    generator.manual_seed(seed)
    epochs = 100
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=1e-3,
    )

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    p = Pipeline(model, criterion, optimizer)
    trainloader = DataLoader(
        dataset=get_cifar10(train=True, transform=train_transform),
        batch_size=64,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
        generator=generator,
        drop_last=False
    )

    testloader = DataLoader(
        dataset=get_cifar10(train=False, transform=test_transform),
        batch_size=1024,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )
    ar = 0.5
    eps = 1/256

    for e in range(epochs):
        model.train()
        train_loss, train_acc = train_epoch_adv(model, trainloader, optimizer, criterion, adv_ratio=ar, epsilon=eps, progress_bar=False)
        model.eval()
        test_loss, test_acc = p.eval_epoch(testloader, False)
        test_adv_loss, test_adv_acc = test_epoch_adv(model, testloader, criterion, adv_ratio=ar, epsilon=eps, progress_bar=False)
        print(
            f"[{e}]"
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Adv Loss: {test_adv_loss:.4f} | "
                        
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Test Acc: {test_acc * 100:.2f}% | "
            f"Test Adv Acc: {test_adv_acc * 100:.2f}%"
            
        )
    
    p.save_checkpoint(save_model_path)
        
    
if __name__ == '__main__':
    main()