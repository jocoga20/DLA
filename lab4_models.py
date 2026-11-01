import torch
import torchvision

def get_vision_model(num_classes):
    m = torchvision.models.resnet18(num_classes=num_classes)
    m.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = torch.nn.Identity()
    return m

def load_trained_model():
    m = get_vision_model(10)
    m.load_state_dict(torch.load('checkpoints/resnet18.cifar10.pt')['model'])
    return m

def load_trained_adv_model():
    m = get_vision_model(10)
    m.load_state_dict(torch.load('adv_model.pt'))
    return m