import math
import sys
import torch
from tqdm import tqdm
import wandb
from model_pipelines import *
from train_utils import *
from utils import *
from metrics import *


def main():
    num_classes = 43
    epochs = 100
    trainloader, testloader = default_loaders(train_batch_size=256, test_batch_size=128)
    model = torchvision.models.resnet50()
    full_fine_tuning(model, num_classes)
    model = model.cuda()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3, weight_decay=1e-3)

    wandb.login(key='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')

    myrun = Run(model=model, optimizer=optimizer, loss=torch.nn.CrossEntropyLoss(reduction='sum'))
    wbrun = wandb.init(entity='jonathangallicoli', project='DLA', name=f'\"fast\" fine tuning', mode='online', dir='wandb_tmp')
    scheduler= torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    chance_level_mean_loss = math.log(num_classes)
    chance_level_accuracy = 1 / num_classes
    print(f'Avg loss: {chance_level_mean_loss}')
    print(f'Avg acc:  {chance_level_accuracy}')
    best_acc = 0.9
    save_path = None
    best_checkpoint = None

    try:
        for e in tqdm(range(epochs)):
            train_loss, train_acc = myrun.train(loader=trainloader)
            scheduler.step()
            with torch.no_grad():
                test_loss, test_acc = myrun.eval(loader=testloader)
            if test_acc > best_acc:
                save_path = f'checkpoints/resnet50.acc{round(test_acc*100)}.pt'
                best_acc = test_acc
                best_checkpoint = {
                    'test_accuracy': best_acc,
                    'epoch': e,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }

            wbrun.log({
                'loss_train': train_loss,
                'acc_train': train_acc,
                'loss_test': test_loss,
                'acc_test': test_acc,
                'loss_diff': loss_diff(train_loss, test_loss),
                'acc_diff': accuracy_diff(train_acc, test_acc),
                'lr': scheduler.get_last_lr()[0]
            })
    finally:
        del trainloader
        del testloader
        if save_path is None:
            print('No progresses')
        else:
            print(f'Saving {save_path}')
            torch.save(best_checkpoint, save_path)
        
        wbrun.finish()

if __name__ == '__main__':
    main()
