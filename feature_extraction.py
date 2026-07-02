from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchvision
from tqdm import tqdm

from system_utils import get_max_workers
from train_utils import Run
from models import *
import torch
import torchvision.transforms.v2 as t

from utils import get_train, gtsrb_mean, gtsrb_std, gtsrb_train_size

@torch.inference_mode()
def get_features(model, loader):
    model.eval()
    model = model.cuda()                  # both model and batch must be on the same device
    features = []
    ys = []

    for x, y in tqdm(loader):
        features.append(model(x.cuda()))
        ys.append(y)

    return torch.vstack(features), torch.concatenate(ys)

def main_save_feature_extraction():
    checkpoint = torch.load('checkpoints/resnet50.acc95.pt')
    m = torchvision.models.resnet50(weights=None)
    rework_model(m, last_layer='linear', output_classes=43, do_freeze_backbone=False)
    m.load_state_dict(checkpoint['model'])
    rework_model(m, last_layer='identity', do_freeze_backbone=True)

    img_size = 128
    my_transform = t.Compose([
        t.Resize((img_size, img_size)),
        t.ToImage(),
        t.ToDtype(torch.float32, scale=True),
        t.Normalize(gtsrb_mean, gtsrb_std)
    ])

    loader = torch.utils.data.DataLoader(
        dataset=get_train(transform=my_transform),
        batch_size=256,                     # i can use bigger batches in test if it runs faster
        shuffle=False,                      # not needed in inference
        num_workers=12,
        pin_memory=True,
    )

    X = get_features(m, loader)
    torch.save(X, f'features/Xy.pt')

def main_classification_head():
    m = get_resnet(last_layer='linear')
    optim = torch.optim.AdamW(params=[p for p in m.parameters() if p.requires_grad], lr=1e-4)
    celoss = torch.nn.CrossEntropyLoss(reduction='sum')
    
    my_transform = t.Compose([
        t.ToImage(), t.ToDtype(torch.float32, scale=True),
        t.Normalize(mean=gtsrb_mean, std=gtsrb_std),
        t.Resize(70),
        t.RandomCrop((64, 64)),		
    ])

    loader = torch.utils.data.DataLoader(
        dataset=get_train(transform=my_transform),
        batch_size=128,                     # i can use bigger batches in test if it runs faster
        shuffle=True,                      # not needed in inference
        num_workers=get_max_workers(),
        pin_memory=True,
        persistent_workers=True
    )

    myrun = Run(model=m, optimizer=optim, loss=celoss)

    for e in tqdm(range(99)):
        myrun.train(loader=loader)

    yp, ys = myrun.train(loader=loader)
    cm = confusion_matrix(ys, yp, normalize='all') * 100
    torch.save(cm, 'cm.pt')
        
if __name__ == '__main__':
    main_save_feature_extraction()