import torch

from backbones import Backbone, ConvNextTiny, DinoV2, MobileNetV3Small, Resnet18
from torch.utils.data import DataLoader, Dataset, TensorDataset
from loggers import PrintLogger, WandbLogger
from train_utils import train
from torch.nn import LazyLinear
from lab1_datasets import get_gtsrb_full_train, get_gtsrb_test

def features_path(backbone_name: str, dataset_name: str):
    return f'features/{backbone_name}_{dataset_name}.pt'

def labels_path(dataset_name: str):
    return f'labels/{dataset_name}.pt'

def feature_extraction(backbones: list[Backbone], datasets_with_name: list[tuple[str, Dataset]]):
    save_label = True
    
    for b in backbones:
        for dataset_name, dataset in datasets_with_name:
            dl = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
            dl.dataset.transform = b.get_transform()

            b.feature_extraction(
                loader=dl,
                x_file=features_path(str(b), dataset_name),
                y_file=(labels_path(dataset_name) if save_label else None),
                progress_bar=True
            )

        save_label = False

def feature_dataset(features_path: str, labels_path: str):
    return TensorDataset(torch.load(features_path), torch.load(labels_path))

def linear_feature_learning(train_features_path, train_labels_path, test_features_path, test_labels_path):
    train_dl = DataLoader(
        dataset=feature_dataset(train_features_path, train_labels_path),
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_dl = DataLoader(
        dataset=feature_dataset(test_features_path, test_labels_path),
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    m = LazyLinear(out_features=43, device='cuda')
    o = torch.optim.SGD(params=m.parameters(), lr=1e-1)
    train(
        model=m,
        optimizer=o,
        train_loader=train_dl,
        eval_loader=test_dl,
        epochs=100,
        max_patience=5,
        logger=WandbLogger(project='DLA', entity='gallicolijonathan-unifi', run_name='gtrsb linear probing')
    )

def main():
    linear_feature_learning('features/convnext_tiny_gtsrb_train.pt', 'labels/gtsrb_train.pt',
                            'features/convnext_tiny_gtsrb_test.pt', 'labels/gtsrb_test.pt')

if __name__ == '__main__':
    main()