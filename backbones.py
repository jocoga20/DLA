from typing import Literal

import torch
from torch.nn import Identity
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models._api import WeightsEnum
from torchvision.transforms import v2 as t
from tqdm import tqdm
from torch.nn import Dropout2d

class Backbone:
    def get_transform(self):
        pass

    def config_for_feature_extraction(self):
        self.model.eval()

    @torch.no_grad()
    def feature_extraction(self,
        loader: DataLoader,
        x_file: str = None,
        y_file: str = None,
        progress_bar: bool = False
    ):
        self.config_for_feature_extraction()
        xs = []
        ys = []

        if progress_bar:
            loader = tqdm(loader, desc=self.name)

        for x, y in loader:
            xs.append(self.model(x.cuda()).cpu())
            ys.append(y)

        xs = torch.cat(xs)
        ys = torch.cat(ys)

        if x_file:
            torch.save(xs, x_file)
        if y_file:
            torch.save(ys, y_file)

        return xs, ys

class TorchvisionModel(Backbone):
    def __init__(self, model_constructor: callable, weights: WeightsEnum = None):
        super().__init__()
        self.model = model_constructor(weights=weights).cuda()
        self.transform = weights.transforms()

    def get_transform(self):
        return self.transform

class Resnet(TorchvisionModel):
    def config_for_feature_extraction(self):
        super().config_for_feature_extraction()
        self.model.fc = Identity()

class Resnet18(Resnet):
    def __init__(self):
        super().__init__(models.resnet18, models.ResNet18_Weights.DEFAULT)

    @staticmethod
    def name(): return 'resnet18'

class MobileNetV3Small(TorchvisionModel):
    def __init__(self):
        super().__init__(models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT)

    def config_for_feature_extraction(self):
        super().config_for_feature_extraction()
        self.model.classifier[3] = Identity()

    @staticmethod
    def name(): return 'mobilenet_v3_small'

class ConvNextTiny(TorchvisionModel):
    def __init__(self):
        super().__init__(models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT)
    
    def config_for_feature_extraction(self):
        super().config_for_feature_extraction()
        self.model.classifier[2] = Identity()

    @staticmethod
    def name(): return 'convnext_tiny'

class DinoV2(Backbone):
    def __init__(self):
        super().__init__()

        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").cuda()

        self.transform = t.Compose([
            t.Resize(256, interpolation=t.InterpolationMode.BICUBIC),
            t.CenterCrop(224),
            t.ToTensor(),
            t.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def get_transform(self):
        return self.transform

    @staticmethod
    def name(): return 'dino_v2'