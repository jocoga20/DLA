
from backbones import ConvNextTiny, DinoV2, MobileNetV3Small, Resnet18
from lab1_datasets import get_gtsrb_full_train, get_gtsrb_test
from torch.utils.data import DataLoader

def feature_extraction(
        batch_size: int,
        num_workers: int,
    ):
    backbones = [Resnet18, MobileNetV3Small, ConvNextTiny, DinoV2]
    for i, b in enumerate(backbones):
        b = b()
        for ds_name, ds_constructor in [('train', get_gtsrb_full_train), ('test', get_gtsrb_test)]:
            dl = DataLoader(
                dataset=ds_constructor(b.get_transform()),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True
            )
            b.feature_extraction(
                loader=dl,
                x_file=f'features/gtrsb_{b.name}_{ds_name}.pt',
                y_file=f'labels/gtrsb_{ds_name}.pt' if i == 0 else None,
                progress_bar=True
            )
            del dl

        del b

feature_extraction(batch_size=64, num_workers=4)