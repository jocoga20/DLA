import torchvision
import torch
import torchvision.transforms.v2 as t
from models import rework_model
from utils import gtsrb_mean, gtsrb_std
from datasets import load_dataset
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import box_iou

class HuggingfaceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.ds = dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        sample = self.ds[index]
        x = self.transform(sample['image'])
        y = {'box': sample['objects']['bbox'], 'class': sample['objects']['category']}
        return x, y

def get_dataset():
    return load_dataset("keremberke/german-traffic-sign-detection", "full")
    

def prepare_backbone():
    m = torchvision.models.resnet50(weights=None)
    rework_model(m, last_layer='linear', output_classes=43, do_freeze_backbone=False)
    m.load_state_dict(torch.load('checkpoints/resnet50.acc95.pt')['model'])
    m = torch.nn.Sequential(*list(m.children())[:-2])
    m.out_channels = 2048       # 2048 if resnet50, 512 if resnet18
    return m

def convert(tensors):
    return tensors[:,2:] + tensors[:,:2]

@torch.inference_mode()
def main():
    anchor_generator = AnchorGenerator(
        sizes=((26,)),
        aspect_ratios=((1.,),)
    )
    model = torchvision.models.detection.FasterRCNN(backbone=prepare_backbone(), num_classes=43, rpn_anchor_generator=anchor_generator)\
    .cuda()
    model.eval()
    model.roi_heads.score_thresh = 0.01

    loader = torch.utils.data.DataLoader(
        dataset=HuggingfaceDataset(get_dataset()['test'], transform=t.Compose([
            t.ToImage(),
            t.ToDtype(torch.float32, scale=True),
            t.Normalize(gtsrb_mean, gtsrb_std),
        ])),
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    for x, y in loader:
        out = model(x.cuda())[0]
        gt_box = torch.tensor(y['box'], device='cuda')
        gt_box = convert(gt_box)
        ioumax = 0
        for box_prop in out['boxes']:
            box_prop = box_prop.reshape((1,-1))
            convert(box_prop)
            print(box_prop, gt_box)
            iou = box_iou(box_prop, gt_box)
            print(iou)
            exit()
            if iou > ioumax:
                ioumax = iou
        print(ioumax)
        break
    
    
if __name__ == '__main__':
    main()
