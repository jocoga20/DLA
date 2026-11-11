from backbones import Resnet18, MobileNetV3Small
from loggers import WandbLogger
from train_utils import train
from torch.optim import AdamW
from torch.utils.data import DataLoader
from lab1_datasets import get_gtsrb, get_gtsrb_train_valid_split
from reproducibility import set_seed

def main(lr, wd, ll):
    rng_seed = 0
    g = set_seed(rng_seed)
    m = Resnet18().fine_tuning_mode(num_classes=43, learnable_layers=ll)
    o = AdamW(params=m.parameters(), lr=lr, weight_decay=wd, fused=True)
    train_ds, valid_ds = get_gtsrb_train_valid_split(split_seed=rng_seed, train_ratio=0.5, train_transform=m.transform, valid_transform=m.transform)
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=g,
        prefetch_factor=4,
        persistent_workers=True,
    )

    valid_dl = DataLoader(
        dataset=valid_ds,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    train(m, o, train_dl, valid_dl, epochs=50, max_patience=3, progress_bars='epoch', logger=WandbLogger(
        project='dla', entity='marcopaglio-universit-di-firenze', run_name=f'resnet {ll} unlocked',
        config_dict={'optimizer': 'adamw', 'lr': lr, 'wd': wd}
    ))

if __name__ == '__main__':
    for learnable_layers in range(3, 5):
        for learning_rate in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
            main(lr=learning_rate, wd=0, ll=learnable_layers)