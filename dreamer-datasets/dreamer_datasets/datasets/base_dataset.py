import os

import torch
from tqdm import tqdm

from ..utils import Timer


class BaseProcessor:
    def __call__(self, data_dict):
        return data_dict

    def process(self, *args, **kwargs):
        pass

    def close(self):
        pass


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, config_path=None, data_path=None, transform=None):
        self._config_path = config_path
        self._data_path = data_path
        self.transform = transform

    @property
    def config_path(self):
        if self._config_path is not None:
            return self._config_path
        elif self._data_path is not None:
            return os.path.join(self._data_path, 'config.json')
        else:
            return None

    @property
    def data_path(self):
        if self._data_path is not None:
            return self._data_path
        elif self._config_path is not None:
            return os.path.dirname(self._config_path)
        else:
            return None

    @classmethod
    def load(cls, data_or_config):
        raise NotImplementedError

    def save(self, save_path, **kwargs):
        raise NotImplementedError

    def open(self):
        pass

    def close(self):
        self._config_path = None
        self._data_path = None
        self.transform = None

    def reset(self):
        self.close()

    def filter(self, *args, **kwargs):
        raise NotImplementedError

    def set_transform(self, transform):
        self.transform = transform

    def process(self, processor, num_workers=0):
        ori_transform = self.transform
        self.transform = processor
        dataloader = torch.utils.data.DataLoader(
            self,
            collate_fn=lambda x: x,
            batch_size=1,
            num_workers=num_workers,
        )
        with Timer('Process Cost'):
            for batch in tqdm(dataloader, total=len(dataloader)):
                for data in batch:
                    if isinstance(data, tuple):
                        processor.process(*data)
                    else:
                        processor.process(data)
            processor.close()
        self.transform = ori_transform

    def __len__(self):
        raise NotImplementedError

    def __del__(self):
        self.close()

    def __getitem__(self, index):
        self.open()
        if isinstance(index, (list, tuple)):
            data_dict = [self._get_data(idx) for idx in index]
        else:
            data_dict = self._get_data(index)
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        return data_dict

    def _get_data(self, index):
        raise NotImplementedError
