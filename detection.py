import torchvision
import torch
from torch import nn
import torchvision.transforms.v2 as t
from models import rework_model
from utils import get_train
from utils import gtsrb_mean, gtsrb_std


class MyBackbone(nn.Module):
    def __init__(self, resnet):
        super().__init__()

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.out_channels = 2048

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return {"0": x}

def prepare_backbone():
    m = torchvision.models.resnet50(weights=None)
    rework_model(m, last_layer='linear', output_classes=43, do_freeze_backbone=False)
    m.load_state_dict(torch.load('checkpoints/resnet50.acc95.pt')['model'])
    rework_model(m, last_layer='identity', do_freeze_backbone=True)
    return MyBackbone(m)

@torch.inference_mode()
def main():
    model = torchvision.models.detection.FasterRCNN(backbone=prepare_backbone(), num_classes=43)
    model.eval()
    mytransform = t.Compose([
        t.Resize((128, 128)),
        t.ToImage(),
        t.ToDtype(torch.float32, scale=True),
        t.Normalize(gtsrb_mean, gtsrb_std)
    ])

    ds = get_train(transform=mytransform)
    loader = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=2,
        shuffle=False,
        pin_memory=True
    )
    x, y = next(iter(loader))
    out = model(x)
    print(out)

if __name__ == '__main__':
    main()