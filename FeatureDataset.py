from dataclasses import dataclass
from typing import Literal


@dataclass
class FeatureDataset:
    backbone_name: str
    def feature_path(self, split: Literal['train', 'test']):
        return f'features/gtrsb_{self.backbone_name}_{split}.pt'

    def label_path(self, split: Literal['train', 'test']):
        return f'labels/gtrsb_{split}.pt'

