from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch.nn import Identity, Linear, Module
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import v2 as t

class Backbone(ABC, Module):
    _hidden_dimension: int = 0
    _max_learnable_layers: int = 0

    def __init__(self, load: bool = True, device: Literal['cuda', 'cpu'] = 'cuda'):
        super().__init__()
        if load:
            self.load(device=device)
        else:
            self.model = None
            self.transform = None

        self.classification_head = None

    @abstractmethod
    def __str__():
        pass

    @abstractmethod
    def _load_model_and_transform(self):
        pass

    @abstractmethod
    def _set_identity_to_old_model_classifier(self):
        pass

    @abstractmethod
    def _unlock_full_classification_layer(self):
        pass

    @abstractmethod
    def _unlock_layers_backwards(self, layers_to_unlock: int):
        pass

    def load(self, device: Literal['cuda', 'cpu'] = 'cuda'):
        self.device = device
        with torch.device(device):
            self._load_model_and_transform()
        self._set_identity_to_old_model_classifier()
        return self

    @torch.inference_mode
    def feature_extraction(self, dataloader: DataLoader, features_path: str = None, labels_path: str = None):
        xs, ys = [], []

        for x, y in dataloader:
            xs.append(self.model(x.to(self.device, non_blocking=True)).cpu())
            ys.append(y)

        xs, ys = torch.cat(xs), torch.cat(ys)
        if features_path:
            torch.save(xs, features_path)
        if labels_path:
            torch.save(ys, labels_path)
        return xs, ys

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = self.classification_head(x)
        return x

    def get_learnable_parameters(self, model: Module):
        if model:
            return [p for p in model.parameters() if p.requires_grad]
        return []

    def parameters(self, recurse = True):
        return self.get_learnable_parameters(self.model) + self.get_learnable_parameters(self.classification_head)
    
    def train(self, mode = True):
        self.model.train(mode)

    def fine_tuning_mode(self, num_classes: int, learnable_layers: int):
        assert 0 <= learnable_layers <= self._max_learnable_layers
        self.freeze_all_layers()
        self._unlock_full_classification_layer()
        if self.classification_head is None:
            self.classification_head = Linear(self._hidden_dimension, num_classes, device=self.device)
        return self
    
    def freeze_all_layers(self):
        self.model.requires_grad_(False)


class TorchvisionModel(Backbone):
    def _load_from_constructor_and_weights(self, constructor, weights):
        self.model = constructor(weights=weights)
        self.transform = weights.transforms()

class ConvNextTiny(TorchvisionModel):
    _hidden_dimension: int = 768
    _max_learnable_layers: int = 8
    def __str__(self): return 'convnext_tiny'

    def _load_model_and_transform(self):
        self._load_from_constructor_and_weights(models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT)

    def _set_identity_to_old_model_classifier(self):
        self.model.classifier[-1] = Identity()

    def _unlock_full_classification_layer(self):
        self.model.classifier.requires_grad_()

    def _unlock_layers_backwards(self, layers_to_unlock: int):
        self.model.features[-layers_to_unlock:].requires_grad_()

class DinoV2(Backbone):
    _hidden_dimension: int = 768
    _max_learnable_layers = 12

    def __str__(): return 'dino_v2'

    def _load_model_and_transform(self):
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.transform = t.Compose([
            t.Resize(256, interpolation=t.InterpolationMode.BICUBIC),
            t.CenterCrop(224),
            t.ToTensor(),
            t.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _set_identity_to_old_model_classifier(self):
        pass

    def _unlock_full_classification_layer(self):
        pass

    def _unlock_layers_backwards(self, layers_to_unlock: int):
        self.model.blocks[-layers_to_unlock:].requires_grad_()


class MobileNetV3Small(TorchvisionModel):
    _hidden_dimension: int = 1024
    _max_learnable_layers = 13
    def __str__(self): return 'mobilenet_v3_small'

    def _load_model_and_transform(self):
        self._load_from_constructor_and_weights(models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.DEFAULT)

    def _set_identity_to_old_model_classifier(self):
        self.model.classifier[-1] = Identity()

    def _unlock_full_classification_layer(self):
        self.model.classifier.requires_grad_()

    def _unlock_layers_backwards(self, layers_to_unlock: int):
        self.model.features[-layers_to_unlock:].requires_grad_()

class Resnet18(TorchvisionModel):
    _hidden_dimension: int = 512
    _max_learnable_layers = 5

    def __str__(): return 'resnet_18'

    def _load_model_and_transform(self):
        self._load_from_constructor_and_weights(models.resnet18, models.ResNet18_Weights.DEFAULT)

    def _set_identity_to_old_model_classifier(self):
        self.model.fc = Identity()

    def _unlock_full_classification_layer(self):
        pass

    def _unlock_layers_backwards(self, layers_to_unlock: int):
        to_unlock = [self.model.layer4, self.model.layer3, self.model.layer2, self.model.layer1]
        to_unlock[layers_to_unlock-1:].requires_grad_()
        if layers_to_unlock == self._max_learnable_layers:
            self.model.conv1.requires_grad_()
            self.model.bn1.requires_grad_()