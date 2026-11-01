from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torchvision.datasets import GTSRB
from lab1_datasets import get_gtsrb_full_train, get_gtsrb_test
from reproducibility import set_seed
from torch.utils.data import Dataset
from PIL.Image import Image

def get_pil_image_size(img: Image):
    return img.width, img.height

def dataset_to_dataframe(dataset: GTSRB):
    data = []
    for x, y in dataset:
        w, h = get_pil_image_size(x)
        data.append({'width': w, 'height': h, 'ratio': w / h, 'label': y})
    return pd.DataFrame(data)

def end_plot(save_path: str = None):
    if save_path:
        plt.savefig(save_path)
        print(f'Image saved @ "{save_path}"')
    else:
        plt.show()

def barplot_params(values):
    uv = [str(x) for x in np.unique(values)]
    bc = np.bincount(values)
    return uv, bc

def plot_discrete_distribution(train_values, test_values, save_path = None):
    uv1, bc1 = barplot_params(train_values)
    uv2, bc2 = barplot_params(test_values)
    plt.clf()
    plt.figure(figsize=(max(len(uv1), len(uv2))/2, 8))
    plt.bar(uv1, bc1, label='train')
    plt.bar(uv2, bc2, label='test')
    plt.legend()
    end_plot(save_path)

def plot_continous_distribution(train_values, test_values, bins, save_path = None):
    plt.clf()
    plt.hist(train_values, bins=bins, label='train')
    plt.hist(test_values, bins=bins, label='test')
    plt.legend()
    end_plot(save_path)

def main():
    df1 = dataset_to_dataframe(get_gtsrb_full_train())
    df2 = dataset_to_dataframe(get_gtsrb_test())
    #plot_discrete_distribution(df1['label'], df2['label'], 'label_distribution.png')
    plot_continous_distribution(df1['ratio'], df2['ratio'], 40, 'ratio_distribution.png')

if __name__ == '__main__':
    main()