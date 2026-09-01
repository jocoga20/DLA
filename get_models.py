import torch
import torchvision.models as models

def get_efficientnet():
	w = models.EfficientNet_B0_Weights.IMAGENET1K_V1
	m = models.efficientnet_b0(weights=w)
	return m, w.transforms()

def get_resnet18():
    w = models.ResNet18_Weights.DEFAULT
    m = models.resnet18(weights=w)
    return m, w.transforms()

def get_resnet34():
	w = models.ResNet34_Weights.DEFAULT
	m = models.resnet34(weights=w)
	return m, w.transforms()

def get_convnext():
	w = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
	m = models.convnext_tiny(weights=w)
	return m, w.transforms()

def get_convnext_large():
	w = models.ConvNeXt_Large_Weights.DEFAULT
	m = models.convnext_large(weights = w)
	return m, w.transforms()

def get_mobilenet():
	w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
	m = models.mobilenet_v3_small(weights=w)
	return m, w.transforms()

def get_resnet152():
	w = models.ResNet152_Weights.DEFAULT
	m = models.resnet152(weights=w)
	return m, w.transforms()

def get_dino():
	from torchvision.transforms import v2
	m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").cuda()
	t = v2.Compose([
		v2.Resize(256),
		v2.CenterCrop(224),
		v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
		v2.Normalize(
			mean=(0.485, 0.456, 0.406),
			std=(0.229, 0.224, 0.225),
		),
	])
	return m, t