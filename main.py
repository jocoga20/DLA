from dataclasses import asdict

from Pipeline import Pipeline
from TrainConfig import TrainConfig
from lab4_datasets import get_cifar10, get_cifar10_train, get_cifar10_valid
from lab4_training import get_loaders, get_vision_model, train_vision_model
from lab4_transforms import train_transform, test_transform
from loggers import WandbLogger
from reproducibility import set_seed
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

def sgd_to_dict(optim: SGD):
    final_dict = {key: optim.param_groups[0][key] for key in ['lr', 'momentum', 'weight_decay', 'dampening', 'nesterov']}
    final_dict['name'] = 'SGD'
    return final_dict

def main():
    rng_seed = 0
    g = set_seed(rng_seed)
    model = get_vision_model(10)
    trainset = get_cifar10(train=True, transform=train_transform)
    testset = get_cifar10(train=True, transform=test_transform)
    optim = SGD(params=model.parameters(), lr=1e-2, momentum=.5, weight_decay=1e-3, nesterov=True, fused=True)
    config = TrainConfig(epochs=40, patience=5, rng_seed=rng_seed, optimizer_config=sgd_to_dict(optim))
    myloss = CrossEntropyLoss(reduction='sum')
    trainloader, testloader = get_loaders(g, trainset, testset)
    mylogger=WandbLogger(project='DLA', run_name='lab4', config_dict=asdict(config))
    Pipeline(model, myloss, optim, logger=mylogger)\
        .train(trainloader, testloader, config.epochs, config.patience, True)\
        .save_checkpoint('checkpoints/resnet18.cifar10.pt')

if __name__ == '__main__':
    main()