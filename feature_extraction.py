from torch.nn import Module
from tqdm import tqdm
from lab1_models import load_resnet18_feature_extraction, load_mobilenet_v3_small_feature_extraction, load_convnext_tiny_feature_extraction, load_dino_v2
from torch.utils.data import DataLoader
import torch
from lab1_datasets import get_gtsrb_full_train, get_gtsrb_test
from torch.utils.data import Dataset

@torch.inference_mode()
def extract_features(model: Module, loader: DataLoader, path: str = None):
    features = torch.cat([model(x.cuda(non_blocking=True)).cpu() for x, _ in loader]).cpu()
    if path:
        torch.save(features, path)
    else:
        return features

def extract_all(names, models, trainset, testset, ds_name, p_bar = False):
    ds_train = get_gtsrb_full_train()
    ds_test = get_gtsrb_test()
    nw = 6
    bs = 64
    dl_train = DataLoader(trainset, batch_size=bs, shuffle=False, pin_memory=True, num_workers=nw, persistent_workers=True)
    dl_test = DataLoader(testset, batch_size=bs, shuffle=False, pin_memory=True, num_workers=nw, persistent_workers=True)

    iterator = zip(names, models)
    if p_bar:
        iterator = tqdm(iterator)

    for name, (transform, model) in iterator:
        ds_train.transform = transform
        ds_test.transform = transform
        model.eval()
        model = model.cuda()
        extract_features(model, dl_train, f'features/{name}_{ds_name}_full_train.pt')
        extract_features(model, dl_test, f'features/{name}_{ds_name}_test.pt')

    del dl_train, dl_test