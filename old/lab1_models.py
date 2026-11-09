from torchvision import models
from torch.nn import Identity
import torch

def load_resnet18_feature_extraction():
    w = models.ResNet18_Weights.IMAGENET1K_V1
    m = models.resnet18(weights=w)
    m.fc = Identity()
    return w.transforms(), m

def load_mobilenet_v3_small_feature_extraction():
    w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    m = models.mobilenet_v3_small(weights=w)
    m.classifier[3] = Identity()
    return w.transforms(), m

def load_convnext_tiny_feature_extraction():
    w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    m = models.convnext_tiny(weights=w)
    m.classifier[2] = Identity()
    return w.transforms(), m

def load_dino_v2_feature_extraction():
    import torch
    from torchvision.transforms import v2 as t

    transform = t.Compose([
        t.Resize(256, interpolation=t.InterpolationMode.BICUBIC),
        t.CenterCrop(224),
        t.ToTensor(),
        t.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model.eval()
    return transform, model

print(load_dino_v2_feature_extraction()[1])