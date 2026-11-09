import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def end_plot(img_path=None):
    if img_path:
        print(f'Saving image "{img_path}"')
        plt.savefig(img_path)
    else:
        plt.show()

def relative_frequency(labels):
    return np.bincount(labels, minlength=43) / len(labels)

def plot_labels(train_labels, test_labels, img_path=None):
    train_labels = relative_frequency(train_labels)
    test_labels = relative_frequency(test_labels)

    plt.plot(train_labels, label='Train')
    plt.plot(test_labels, label='Test')
    plt.axhline(1/43, label='Balanced frequency', color='black', linestyle='--')
    plt.xticks(range(0, 43, 2))
    plt.legend()
    plt.ylabel('Relative frequencies')
    plt.xlabel('Classes')
    plt.tight_layout()
    end_plot(img_path)