from matplotlib import pyplot as plt
import numpy as np
import torch
from lab4_transforms import test_transform, revert_test_transform
from adversarial import targeted_fgsm
from lab4_datasets import get_cifar10, class_names
from lab4_models import load_trained_adv_model, load_trained_model
from torch.nn import CrossEntropyLoss

def batch_from_indices(indices: list[int]):
    ds = get_cifar10('test', test_transform)
    xs = []
    ys = []
    for i in indices:
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    # stack crea una nuova dimensione, che si posizione in posizione dim
    # cat invece concatena sulla dimensione specificata
    return torch.stack(xs, dim=0).cuda(), torch.tensor(ys)

def draw_sample_model_row_by_batch(xs, model, axes, row):
    softmaxed = model(xs).softmax(dim=1).max(dim=1)
    classes = softmaxed.indices
    probs = softmaxed.values
    for i in range(3):
        axes[row, i].imshow(revert_test_transform(xs[i]))
        axes[row, i].set_xticks([])
        axes[row, i].set_yticks([])
        axes[row, i].set_title(class_names[classes[i]])
        axes[row, i].set_xlabel(round(probs[i].item() * 100, 2))

def end_draw(save_path: str = None):
    if save_path:
        plt.savefig(save_path)
        print(f'Image saved @ "{save_path}"')
    else:
        plt.show()

def draw_samples_batches_and_models(x1, x2, m1, m2, img_side_px, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(3*img_side_px, 2*img_side_px, 'px'))
    fig.text(.5, .95, 'Model', fontsize=16, ha='center')
    fig.text(.5, .5, 'Model ADV', fontsize=16, ha='center')
    draw_sample_model_row_by_batch(x1, m1, axes, 0)
    draw_sample_model_row_by_batch(x2, m2, axes, 1)
    end_draw(save_path)

def draw_samples_by_indices(indices: list[int], img_side_px = 600, save_path: str = None):
    xs, _ = batch_from_indices(indices)
    m1 = load_trained_model().cuda()
    m2 = load_trained_adv_model().cuda()
    m1.eval()
    m2.eval()
    draw_samples_batches_and_models(xs, xs, m1, m2, img_side_px, save_path)

def draw_tFGSM_samples_by_indices(indices: list[int], target_ys: torch.Tensor, epsilon: float, img_side_px: int = 600, save_path: str = None):
    m1 = load_trained_model().cuda()
    m2 = load_trained_adv_model().cuda()
    m1.eval()
    m2.eval()
    # targeted_fgsm handles the copy and detach from graph
    xs, _ = batch_from_indices(indices)
    criterion = CrossEntropyLoss(reduction='sum')
    x1 = targeted_fgsm(xs.cuda(non_blocking=True), target_ys, m1, epsilon=epsilon, criterion=criterion)
    x1_adv = targeted_fgsm(xs.cuda(non_blocking=True), target_ys, m2, epsilon=epsilon, criterion=criterion)

    draw_samples_batches_and_models(x1, x1_adv, m1, m2, img_side_px, save_path)