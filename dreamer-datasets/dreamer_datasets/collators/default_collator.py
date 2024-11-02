import numpy as np
import torch

from .. import utils


class DefaultCollator:
    def __call__(self, batch):
        batch_dict = dict()
        if isinstance(batch, list):
            for key in batch[0]:
                batch_dict[key] = self._collate([d[key] for d in batch])
        elif isinstance(batch, dict):
            for key in batch:
                batch_dict[key] = self._collate(batch[key])
        else:
            assert False
        return batch_dict

    def _collate(self, batch):
        if isinstance(batch, (list, tuple)):
            if isinstance(batch[0], torch.Tensor):
                batch = utils.stack_data(batch)
            elif isinstance(batch[0], np.ndarray):
                batch = utils.stack_data(batch)
                batch = torch.from_numpy(batch)
            elif isinstance(batch[0], (np.bool_, np.number, np.object_)):
                batch = torch.as_tensor(batch)
            elif isinstance(batch[0], dict):
                batch = {key: self._collate([d[key] for d in batch]) for key in batch[0]}
            elif isinstance(batch[0], (list, tuple)):
                # batch = type(batch[0])([self._collate(d) for d in zip(*batch)])
                batch = type(batch[0])([self._collate(d) for d in batch])
        elif isinstance(batch, np.ndarray):
            batch = torch.from_numpy(batch)
        return batch
    