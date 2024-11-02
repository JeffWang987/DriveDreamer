import math

import numpy as np
import torch


class DefaultSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size=None, shuffle=True, infinite=True, seed=6666):
        self.shuffle = shuffle
        self.infinite = infinite
        self.seed = seed
        self.epoch = 0
        self.data_size = len(dataset)
        if batch_size is not None:
            self.total_size = int(math.ceil(self.data_size / batch_size)) * batch_size
        else:
            self.total_size = self.data_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.total_size

    def __iter__(self):
        while True:
            np.random.seed(self.seed + self.epoch)
            self.epoch += 1
            indices = np.zeros((0,), dtype=np.int64)
            while len(indices) < self.total_size:
                indices_i = np.arange(self.data_size)
                if self.shuffle:
                    indices_i = np.random.permutation(indices_i)
                num_data = min(len(indices_i), self.total_size - len(indices))
                indices = np.hstack((indices, indices_i[:num_data]))
            yield from indices
            if not self.infinite:
                break
