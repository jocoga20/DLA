import torchvision
from tqdm import tqdm

from system_utils import get_max_workers
from train_utils import Run
from model_pipelines import *
import torch
import torchvision.transforms.v2 as t

from utils import get_test, get_train, gtsrb_mean, gtsrb_std, gtsrb_train_size
import torchvision.models as vmodels

@torch.inference_mode()
def get_features(model, loader: torch.utils.data.DataLoader):
    model.eval()
    features = []
    ys = []

    for x, y in tqdm(loader):
        features.append(model(x.cuda()))
        ys.append(y)

    features = torch.vstack(features)
    features = torch.nn.functional.normalize(features, dim=1)

    return features, torch.concat(ys)

def nearest_centroid_y(centroids, x):
    return (centroids - x).abs().sum(axis=1).argmin()

def backbone_accuracy(model, trainloader, testloader):
    train_feats, ytrain = get_features(model, trainloader)
    test_feats, ytest = get_features(model, testloader)
    test_size = test_feats.shape[0]
    train_centroids = torch.vstack([train_feats[ytrain == i].mean(axis=0) for i in range(43)])
    
    corrects = 0

    for i in range(test_size):
        x = test_feats[i]
        y = ytest[i]

        k = nearest_centroid_y(train_centroids, x)
        if k == y:
            corrects += 1
    
    return corrects / test_size * 100

def main_several_models():
    models = [vmodels.resnet18, vmodels.resnet34, vmodels.resnet50, vmodels.resnet101, vmodels.resnet152]
    weights = [vmodels.ResNet18_Weights, vmodels.ResNet34_Weights, vmodels.ResNet50_Weights, vmodels.ResNet101_Weights, vmodels.ResNet152_Weights]

    for m, w in zip(models, weights):
        print(m.__name__)
        m = m(weights=w.DEFAULT)
        feature_extraction(m)
        my_transform = w.DEFAULT.transforms()
        bs = 512

        trainloader = torch.utils.data.DataLoader(
            dataset=get_train(transform=my_transform),
            batch_size=bs,
            shuffle=False,
            num_workers=get_max_workers() // 2,
            pin_memory=True,
            persistent_workers=False
        )

        testloader = torch.utils.data.DataLoader(
            dataset=get_test(transform=my_transform),
            batch_size=bs,
            shuffle=False,
            num_workers=get_max_workers() // 2,
            pin_memory=True,
            persistent_workers=False
        )

        print(backbone_accuracy(m, trainloader, testloader))

def main():
    m = torchvision.models.resnet50(weights=None).cuda()
    rework_model(m, last_layer='linear', output_classes=43, do_freeze_backbone=False)
    train_dict = torch.load('checkpoints/resnet50.acc95.pt')
    m.load_state_dict(train_dict['model'])
    rework_model(m, last_layer='identity', do_freeze_backbone=True)

    img_size = 128
    my_transform = t.Compose([
        t.Resize((img_size, img_size)),
        t.ToImage(),
        t.ToDtype(torch.float32, scale=True),
        t.Normalize(gtsrb_mean, gtsrb_std)
    ])

    bs = 1024

    trainloader = torch.utils.data.DataLoader(
        dataset=get_train(transform=my_transform),
        batch_size=bs,                     # i can use bigger batches in test if it runs faster
        shuffle=False,                      # not needed in inference
        num_workers=get_max_workers() // 2,
        pin_memory=True,
        persistent_workers=False
    )

    testloader = torch.utils.data.DataLoader(
        dataset=get_test(transform=my_transform),
        batch_size=bs,
        shuffle=False,
        num_workers=get_max_workers() // 2,
        pin_memory=True,
        persistent_workers=False
    )
    print(backbone_accuracy(m, trainloader, testloader))

if __name__ == '__main__':
    main()