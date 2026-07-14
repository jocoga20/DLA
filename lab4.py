from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
import torch
import torchvision
import torchvision.transforms.v2 as t
from torch import softmax
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from Training import *
from model_pipelines import full_fine_tuning

def save_checkpoint(filename, model, optimizer):
    torch.save({'model': model, 'optimizer': optimizer}, filename)

@torch.inference_mode()
def max_softmax_eval(model, loader, temp = 1):
    values = []

    for x, y in tqdm(loader):
        logits = model(x.cuda())
        values.append(logits)
        
    return torch.concat(values)
    

def setup_model(checkpoint = None):
    m = torch.load(checkpoint, weights_only=False)['model']
    m.fc = torch.nn.Identity()
    return m

@torch.inference_mode()
def get_id_centroid(model, id_loader):
    mean_tensor = torch.zeros(size=(2048,)).cuda()
    model.eval()

    for x, y in tqdm(id_loader):
        mean_tensor += model(x.cuda()).sum(dim=0).squeeze()
    
    return mean_tensor / 9000


def mean(x): return sum(x) / len(x)

def dist(x, y):
    return ((x - y) ** 2).sum()

def present(x):
    print(x.mean(), x.std())

def get_centroids(x, y):
    return torch.vstack([x[y == t].mean(0) for t in np.unique(y)])

def centroids_distance(centroids: torch.Tensor, points: torch.Tensor):
    return torch.cdist(points, centroids).min(dim=1).values


default_transform = t.Compose([t.ToImage(), t.ToDtype(torch.float32, scale=True)])

from torchvision.transforms.v2 import InterpolationMode

train_transform = t.Compose([
    t.AugMix(),
    t.ToImage(),
    t.ToDtype(torch.float32, scale=True)
])

infer_transform = t.Compose([
    t.ToImage(),
    t.ToDtype(torch.float32, scale=True)
])

# train_transform = t.Compose([
#     t.Resize(256, interpolation=InterpolationMode.BILINEAR),
#     t.RandomResizedCrop(224, scale=(0.8, 1.0)),
#     t.RandomHorizontalFlip(p=0.5),
#     t.RandAugment(),
#     t.ToImage(),
#     t.ToDtype(torch.float32, scale=True),
#     t.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ),
#     t.RandomErasing(
#         p=0.25,
#         scale=(0.02, 0.2),
#         ratio=(0.3, 3.3)
#     ),
# ])

# test_transform = t.Compose([
#     t.Resize(256, interpolation=InterpolationMode.BILINEAR),
#     t.CenterCrop(224),
#     t.ToImage(),
#     t.ToDtype(torch.float32, scale=True),
#     t.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     ),
# ])

def get_id_dataset(train, transform):
    return torchvision.datasets.CIFAR10('./mydatasets', train=train, transform=transform, download=True)
    
def get_ood_dataset(transform=default_transform, size=50_000):
    return torchvision.datasets.FakeData(size = size, image_size=(3, 32, 32), transform=transform, num_classes=1)

def setup_model():
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
    full_fine_tuning(model, output_classes=10)
    return torch.compile(model.cuda())

def main_train_model():
    torch.set_float32_matmul_precision("high")
    model = setup_model()
    optim = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs = 100
    patience = 0
    test_acc_inc = 0.001
    max_patience = 10

    training = Training(
        model=model,
        optimizer=optim,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    ).save_best_model('resnet50.pt')

    trainloader = DataLoader(
        dataset=get_id_dataset(train=True, transform=train_transform),
        batch_size=128,
        shuffle=True,
        num_workers=8,# todo: com è fatto state dict di model, optim e scheds
        pin_memory=True,
        persistent_workers=True
    )

    testloader = DataLoader(
        dataset=get_id_dataset(train=False, transform=train_transform),
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    training.train(trainloader=trainloader, testloader=testloader, epochs=epochs, progress_bar=True)

class XDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

def main_save_max_softmax():
    x_ood = torch.load('x_ood.pt')
    x_train = torch.load('x_train.pt')
    x_test = torch.load('x_test.pt')

    default_loader = lambda dataset: DataLoader(
        dataset=XDataset(dataset),
        batch_size=256,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )

    ood_loader = default_loader(x_ood)
    id_train_loader = default_loader(x_train)
    id_test_loader = default_loader(x_test)


    model = torch.nn.Linear(2048, 10, device='cuda')
    model.load_state_dict(torch.load('resnet50.acc89.pt'))
    model.eval()

    temperature = 10

    ms_ood = max_softmax_eval(model, ood_loader, temp=temperature)
    ms_id_train = max_softmax_eval(model, id_train_loader, temp=temperature)
    ms_id_test = max_softmax_eval(model, id_test_loader, temp=temperature)

    present(ms_ood)
    present(ms_id_train)
    present(ms_id_test)
    

def precision_recall(positives, negatives, threshold):
    tp = tn = fp = fn = 0

    for p in positives:
        if p > threshold:
            fn += 1
        else:
            tp += 1
    
    for n in negatives:
        if n > threshold:
            tn += 1
        else:
            fp += 1

    return tp / (tp + fp), tp / (tp + fn)

def main_detection_threshold():
    ms_id = torch.load('ms_id.pt').cpu().numpy()
    ms_ood = torch.load('ms_ood.pt').cpu().numpy()

    for t in np.linspace(0.6, 0.7, 10):
        print(t)
        p, r = precision_recall(ms_ood, ms_id, t)
        print(f'{2*p*r/(p+r)}, {p}, {r}')

def main():
    model = setup_model()
    train_id, train_ood = split_id_ood(get_dataset(train=True))
    test_id, test_ood = split_id_ood(get_dataset(train=False))

    # training
    train_loader = torch.utils.data.DataLoader(
        dataset=train_id,
        batch_size=256,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_id,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
        momentum=0.,
        weight_decay=1e-4,
    )

    myloss = torch.nn.CrossEntropyLoss(reduction='sum')
    epochs = 10

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, myloss)
        test_loss, test_acc = eval(model, test_loader, myloss)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"train loss={train_loss:.4f} "
            f"test loss={test_loss:.4f} "
            f"train acc={train_acc:.4f} "
            f"test acc={test_acc:.4f}"
        )

        save_checkpoint('check.pt', model, optimizer)

if __name__ == '__main__':
    main_train_model()