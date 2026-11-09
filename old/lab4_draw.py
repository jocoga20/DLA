import math
import random

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from tqdm import tqdm

from adversarial import get_gradient
from lab4_transforms import revert_test_transform
from reproducibility import set_seed


def draw_samples(sample_index, eps_ints, x_batch):
    ncols = 8
    nrows = math.ceil((eps_ints + 1) / ncols)

    img_side_size = 5

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(img_side_size * ncols, img_side_size * nrows),
        squeeze=False
    )

    for i in range(eps_ints + 1):
        xi = x_batch[i]

        img_xi = revert_test_transform(xi)

        axes.flat[i].imshow(img_xi)
        axes.flat[i].axis('off')
        axes.flat[i].set_title(
            f'{i}/255',
            fontsize=40
        )

    for i in range(eps_ints + 1, nrows * ncols):
        axes.flat[i].axis('off')

    fig.tight_layout()

    fig.savefig(
        f'samples/image_x{sample_index}.pdf'
    )
    plt.close(fig)

def draw_probs(sample_index, probs_batch):
    save_path = f'samples/probs_x{sample_index}.pdf'
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f'sample #{sample_index}')

    for i in range(10):
        ax.plot(probs_batch[:, i], label=f'Class {i}')

    ax.legend()
    ax.set_xlabel('i')
    ax.set_ylabel('P(y|x_i)')

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def draw_all(x, y, model, sample_index, eps_ints):
    g_sign = get_gradient(x, y, model).sign()
    eps = 1 / 255
    x_batch = torch.cat([x] + [x + eps * i * g_sign for i in range(1, eps_ints + 1)])
    probs_batch = torch.softmax(model(x_batch), dim=1).detach().cpu()
    draw_probs(sample_index, probs_batch)
    draw_samples(sample_index, eps_ints, x_batch)

def k_samples_with_perturbation(model: Module, dataset: Dataset, k: int, eps_ints: int):
    set_seed(0)
    model.eval()
    ds_len = len(dataset)
    sample_indices = random.sample(range(ds_len), k)
    for i in tqdm(sample_indices):
        x, y = dataset[i]
        x = x.cuda().unsqueeze(0)
        y = torch.tensor(y, device='cuda').unsqueeze(0)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('x_i')
        plt.ylabel('P(y|x_i)')
        plt.grid()
        draw_all(x, y, model, i, eps_ints)