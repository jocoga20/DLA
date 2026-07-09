from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms.v2 as t
from torch import softmax
from tqdm import tqdm

mytransform = t.Compose([t.ToImage(), t.ToDtype(torch.float32, scale=True)])

def get_dataset(train):
    return torchvision.datasets.FashionMNIST('./mydatasets', download=True, train=train, transform=mytransform)

def split_id_ood(dataset):
    ood_idx = []
    id_idx = []

    for i, (_, y) in enumerate(tqdm(dataset)):
        if y == 8:
            ood_idx.append(i)
        else:
            id_idx.append(i)

    return torch.utils.data.Subset(dataset, id_idx), torch.utils.data.Subset(dataset, ood_idx)

def save_checkpoint(filename, model, optimizer):
    torch.save({'model': model, 'optimizer': optimizer}, filename)


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
    return test_loss,test_acc

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
    print(x.min(), x.mean(), x.max())

@torch.inference_mode()
def main_detect_ood():
    model = setup_model('check.pt').cuda()
    model.eval()

    id_centroid = torch.load('id_centroid.pt', weights_only=False)

    id_train, ood_train = split_id_ood(get_dataset(train=True))

    id_loader = torch.utils.data.DataLoader(id_train, batch_size=1024, shuffle=False, pin_memory=True)

    ood_loader = torch.utils.data.DataLoader(ood_train, batch_size=1024, shuffle=False, pin_memory=True)

    id_train = torch.vstack([model(x.cuda()).cpu() for x, y in tqdm(id_loader)]).cuda()
    ood_train = torch.vstack([model(x.cuda()).cpu() for x, y in tqdm(ood_loader)]).cuda()

    id_dist = ((id_train - id_centroid) ** 2).sum(dim=1)
    ood_dist = ((ood_train - id_centroid) ** 2).sum(dim=1)

    print('ID')
    present(id_dist)
    print('OOD')
    present(ood_dist)

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