import torch
from torch.utils.data import Dataset

import numpy as np


class BagOfDatasets(Dataset):
    r"""Wrapper class over several dataset classes.
    We assume each dataset returns ONLY images (no labels).
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.n = len(datasets)

    def __len__(self):
        lengths = [len(dataset) for dataset in self.datasets]
        return max(lengths)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image1, image2, ...)
        """
        items = []
        for dataset in self.datasets:
            item = dataset.__getitem__(index)
            items.append(item)

        return items


def sample_minibatch_from_cache(cache, batch_size, sample_size):
    replace_sample = True if (len(cache) < sample_size) else False

    minibatch = []
    for i in range(batch_size):
        indexes = np.random.choice( np.arange(len(cache)),
                                    replace=replace_sample, 
                                    size=sample_size)
        items = cache[indexes]
        minibatch.append(items)
    minibatch = np.stack(minibatch)
    minibatch = torch.from_numpy(minibatch)
    return minibatch


def sample_minibatch(dataset, batch_size, sample_size):
    minibatch = []
    replace_sample = True if (len(dataset) < sample_size) else False
    for i in range(batch_size):
        items = []
        indexes = np.random.choice( np.arange(len(dataset)),
                                    replace=replace_sample, 
                                    size=sample_size)
        for index in indexes:
            item = dataset.__getitem__(index)
            if isinstance(item, tuple):
                item = item[0]
            items.append(item)

        items = torch.stack(items)
        minibatch.append(items)

    minibatch = torch.stack(minibatch)
    return minibatch


def normal_init(m, mean=0., std=0.01):
    m.weight.data.normal_(mean, std)
    