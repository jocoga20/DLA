from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay
import torchvision

from lab4 import get_id_dataset, get_ood_dataset, max_softmax_eval, present
import torch
from torch.utils.data import DataLoader

@torch.inference_mode()
def save_logits():
    default_loader = lambda dataset: DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    train_loader = default_loader(get_id_dataset(train=True))
    test_loader = default_loader(get_id_dataset(train=False))
    ood_loader = default_loader(get_ood_dataset())

    model = torchvision.models.resnet18(weights=None).cuda()
    model.fc = torch.nn.Linear(model.fc.in_features, 10, device='cuda')
    model.load_state_dict(torch.load('resnet50.acc96.pt'))
    model.eval()

    t = 1
    ms_ood = max_softmax_eval(model, ood_loader, temp=t)
    torch.save(ms_ood, 'logits_ood.pt')
    ms_id_train = max_softmax_eval(model, train_loader, temp=t)
    torch.save(ms_id_train, 'logits_id_train.pt')
    ms_id_test = max_softmax_eval(model, test_loader, temp=t)
    torch.save(ms_id_test, 'logits_id_test.pt')

def max_softmax(x):
    return torch.softmax(x, dim=1).max(dim=1).values.cpu()

def ood_detection():
    log_ood = torch.load('logits_ood.pt')
    log_id_train = torch.load('logits_id_train.pt')

    pred_ood = max_softmax(log_ood)
    pred_id_train = max_softmax(log_id_train)
    x = np.concat([pred_ood, pred_id_train])
    y = np.concat([np.zeros(50_000), np.ones(50_000)])
    disp = RocCurveDisplay.from_predictions(y, x)
    disp.ax_.set_title('PR')
    disp.ax_.grid(True)
    plt.savefig('pr.png')

if __name__ == '__main__':
    ood_detection()