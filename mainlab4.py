import optuna
import torch
from tqdm import tqdm
from TrainConfig import TrainConfig
from adversarial import eval_epoch_adv, train_epoch_adv
from lab4_datasets import get_cifar10_train, get_cifar10_valid, get_cifar10
from lab4_training import get_loaders, get_vision_model, train_vision_model
from lab4_transforms import train_transform, test_transform
from loggers import WandbLogger
from reproducibility import set_seed
from train_utils import eval_epoch, train

def hyperparameter_tuning(logfile: str):
    optuna.logging.set_verbosity(optuna.logging.WARN)
    study = optuna.create_study(direction='maximize')
    study.optimize(evaluate_config, n_trials=30, show_progress_bar=True)
    sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    with open(logfile, 'w') as file:
        for t in sorted_trials:
            file.write(f'{t.value} {t.params}\n')

def evaluate_config():
    trainconfig = TrainConfig(
        epochs=150,
        learning_rate=.1,
        optimizer='sgd',
        weight_decay=5e-4,
    )

    rng_seed = 0
    g = set_seed(rng_seed)
    model = get_vision_model(10)
    train_vision_model(
        model=model,
        trainset=get_cifar10(True, train_transform),
        validset=get_cifar10(False, test_transform),
        config=trainconfig,
        logger=WandbLogger(project='DLA', run_name='lab4'),
        generator=g
    )
    torch.save(model.state_dict(), 'resnet18.cifar10.pt')

def main_adversarial_training():
    config = TrainConfig(epochs=150, learning_rate=1e-3, optimizer='sgd', weight_decay=5e-4)
    rng_seed = 0
    g = set_seed(rng_seed)
    model = get_vision_model(10)
    train_loader, valid_loader = get_loaders(generator=g, trainset=get_cifar10_train(train_transform), validset=get_cifar10_valid(test_transform))
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, fused=True)
    celoss = torch.nn.CrossEntropyLoss(reduction='sum')
    logger = WandbLogger(project='DLA', run_name='lab4 adv train')
    for e in tqdm(range(config.epochs)):
        train_loss, train_acc = train_epoch_adv(model, train_loader, optimizer, celoss, adv_ratio=.5, epsilon=1e-3)
        valid_loss, valid_acc = eval_epoch(model, valid_loader, celoss)
        valid_adv_loss, valid_adv_acc = eval_epoch_adv(model, valid_loader, celoss, adv_ratio=.5, epsilon=1e-3)
        logger.log(step=e, data_dict={'train_loss': train_loss, 'valid_loss': valid_loss, 'valid_adv_loss': valid_adv_loss, 'train_acc': train_acc, 'valid_acc': valid_acc, 'valid_adv_acc': valid_adv_acc})
    logger.end_run()

if __name__ == '__main__':
    main_adversarial_training()