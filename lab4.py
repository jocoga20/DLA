from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.v2 as t
from torch import softmax
from tqdm import tqdm

def save_checkpoint(filename, model, optimizer):
    torch.save({'model': model, 'optimizer': optimizer}, filename)

@torch.no_grad()
def eval(model, test_loader, myloss):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            y = y.cuda(non_blocking=True)
            y = torch.where(y > 8, y - 1, y)

            logits = model(x.cuda(non_blocking=True))

            loss = myloss(logits, y)

            running_loss += loss.item()

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    test_loss = running_loss / total
    test_acc = correct / total
    return test_loss, test_acc

def train(model, train_loader, optimizer, myloss):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        y = y.cuda(non_blocking=True)
        y = torch.where(y > 8, y - 1, y)

        optimizer.zero_grad()

        logits = model(x.cuda(non_blocking=True))

        loss = myloss(logits, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_loss = running_loss / total
    train_acc = correct / total
    return train_loss, train_acc

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
cifar10_transform = t.Compose([t.ToImage(), t.ToDtype(torch.float32, scale=True), t.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_id_dataset(train):
    return torchvision.datasets.CIFAR10('./mydatasets', train=train, transform=cifar10_transform, download=True)

def get_ood_dataset(train):
    return torchvision.datasets.FakeData('./mydatasets', train=train, transform=default_transform, size=10_000 if train else 5_000, image_size=(3, 32, 32), num_classes=1)

def main_detect_ood():
    id_train, id_test = get_id_dataset(train=True), get_id_dataset(train=False)
    id_train_loader = torch.utils.data.DataLoader(
        dataset=id_train,
        batch_size=1024,
        shuffle=True,
        pin_memory=True
    )

    id_test_loader = torch.utils.data.DataLoader(
        dataset=id_test,
        batch_size=2048,
        shuffle=False,
        pin_memory=True
    )

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=1e-6)
    myloss = torch.nn.CrossEntropyLoss(reduction='sum')
    best_model_dict = None
    best_acc = 0

    for e in range(50):
        train_loss, train_acc = train(model, id_train_loader, opt, myloss)
        test_loss, test_acc = eval(model, id_test_loader, myloss)
        if test_acc > best_acc:
            best_model_dict = model.state_dict()
            best_acc = test_acc

        print(f'[{e}] L: {train_loss:.4f} | {test_loss:.4f} A: {train_acc:.2f} | {test_acc:.2f}')

    torch.save(best_model_dict, f'resnet18.acc{int(best_acc*100)}.pt')

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
    main_detect_ood()