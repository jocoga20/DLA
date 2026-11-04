from lab1_plots import plot_labels
from lab1_datasets import get_gtsrb_full_train, get_gtsrb_test
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

def get_labels_from_dataset(dataset: Dataset):
    return [y for _, y in tqdm(dataset)]

def plot_and_relative_balance():
    print('co')
    train_labels = get_labels_from_dataset(get_gtsrb_full_train())
    print('coc')
    test_labels = get_labels_from_dataset(get_gtsrb_test())
    plot_labels(train_labels, test_labels, 'labels.png')

    train_labels = np.bincount(train_labels)
    test_labels = np.bincount(test_labels)

    max_class_train = max(train_labels)
    min_class_train = min(train_labels)

    max_class_test = max(test_labels)
    min_class_test = min(test_labels)

    print(f'Train: {max_class_train / min_class_train}')
    print(f'Test: {max_class_test / min_class_test}')

plot_and_relative_balance()