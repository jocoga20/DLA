from sklearn.model_selection import KFold
import torch
import torchvision
from tqdm import tqdm
from Pipeline import Pipeline
from TrainConfig import TrainConfig
from loggers import NoLogger, PrintLogger, WandbLogger
from reproducibility import set_seed
from train_utils import train
from lab4_transforms import train_transform, test_transform
from torch.utils.data import DataLoader, Dataset, Subset
from torch.nn import BatchNorm2d, Conv2d, Flatten, Linear, MaxPool2d, Module, ReLU, Sequential
from torch.optim import lr_scheduler

def get_loaders(generator: torch.Generator, trainset: Dataset, validset: Dataset):
    nwtot = 12
    nwtrain = 8
    trainloader=DataLoader(
        dataset=trainset,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        num_workers=nwtrain,
        persistent_workers=False,
        generator=generator
    )

    validloader=DataLoader(
        dataset=validset,
        batch_size=64,
        shuffle=False,
        pin_memory=True,
        num_workers=nwtot - nwtrain
    )
    return trainloader, validloader

def count_params(model: Module):
    return sum(p.numel() for p in model.parameters())

def get_trainable_parameters(model: Module):
    return [p for p in model.parameters() if p.requires_grad]

def train_vision_model(
    model: Module,
    trainset: Dataset,
    validset: Dataset,
    config: TrainConfig,
    generator: torch.Generator,
    logger: NoLogger = None
):

    if config.optimizer == 'sgd':
        optim = torch.optim.SGD(
            params=get_trainable_parameters(model),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True
        )
    else:
        optim = torch.optim.AdamW(
            params=get_trainable_parameters(model),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

    logger = logger or NoLogger()
    trainloader, testloader = get_loaders(generator, trainset, validset)
    scheduler = lr_scheduler.CosineAnnealingLR(optim, config.epochs, eta_min=1e-7)
    best_acc = train(model, optim, trainloader, testloader, config, scheduler=scheduler, logger=logger, progress_bars='epoch')
    logger.end_run()

    del trainloader, testloader
    return best_acc

def cross_validation(model: Module, dataset: Dataset, config: TrainConfig, rng_seed: int, progress_bar: bool = False, k: int = 5):
    itersplit = KFold(n_splits=k, shuffle=True, random_state=rng_seed).split(dataset)
    if progress_bar:
        itersplit = tqdm(itersplit, total=k)

    accs = []
    for train_idx, valid_idx in itersplit:
        accs.append(train_vision_model(
            model=model,
            trainset=Subset(dataset, train_idx),
            validset=Subset(dataset, valid_idx),
            config=config,
            rng_seed=rng_seed
        ))
    return accs