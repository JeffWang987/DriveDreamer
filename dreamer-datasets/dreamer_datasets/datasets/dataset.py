import logging
import os

from .. import utils
from .base_dataset import BaseDataset
from .lmdb_dataset import LmdbDataset
from .pkl_dataset import PklDataset


def get_abs_path(abs_or_rel_path, data_dir=None):
    if os.path.isabs(abs_or_rel_path):
        assert os.path.exists(abs_or_rel_path)
        return abs_or_rel_path
    else:
        data_dirs = [data_dir] if data_dir is not None else []
        data_dirs.append(utils.get_data_dir())
        for data_dir in data_dirs:
            abs_path = os.path.join(data_dir, abs_or_rel_path)
            if os.path.exists(abs_path):
                return abs_path
    raise IOError('{} is not exist'.format(abs_or_rel_path))


def get_rel_path(abs_path, data_dir=None):
    data_dirs = [data_dir] if data_dir is not None else []
    data_dirs.append(utils.get_data_dir())
    for data_dir in data_dirs:
        if abs_path.startswith(data_dir):
            rel_path = abs_path[len(data_dir) :]
            if rel_path[0] == '/':
                rel_path = rel_path[1:]
            return rel_path
    return abs_path


def sort_paths(paths):
    new_paths = []
    idx = 0
    for path in paths:
        config = load_config(path)
        key_names = config['_key_names']
        if 'data_index' in key_names:
            new_paths.insert(idx, path)
            idx += 1
        else:
            new_paths.append(path)
    return new_paths


def load_config(data_or_config):
    if isinstance(data_or_config, str):
        if os.path.isdir(data_or_config):
            config_path = os.path.join(data_or_config, 'config.json')
        else:
            config_path = data_or_config
        config = utils.load_file(config_path)
        config['config_path'] = config_path
        if 'data_path' not in config:
            config['data_path'] = os.path.dirname(config_path)
        else:
            config['data_path'] = get_abs_path(config['data_path'])
    elif isinstance(data_or_config, dict):
        config = data_or_config
    else:
        assert False
    return config


def load_dataset(data_or_config):
    if isinstance(data_or_config, list):
        return ConcatDataset.load(data_or_config)
    dataset_dict = {
        'Dataset': Dataset,
        'LmdbDataset': LmdbDataset,
        'PklDataset': PklDataset,
    }
    config = load_config(data_or_config)
    class_name = config['_class_name']
    return dataset_dict[class_name].load(config)


class Dataset(BaseDataset):
    def __init__(self, datasets):
        super(Dataset, self).__init__()
        assert len(datasets) > 0
        self.datasets = datasets

    @classmethod
    def load(cls, data_or_config):
        if isinstance(data_or_config, list):
            return ConcatDataset.load(data_or_config)
        config = load_config(data_or_config)
        class_name = config['_class_name']
        if class_name != 'Dataset':
            return load_dataset(config)
        data_path = config['data_path']
        config_paths = config['config_paths']
        config_paths = [get_abs_path(config_path, data_path) for config_path in config_paths]
        config_paths = sort_paths(config_paths)
        datasets = [load_dataset(config_path) for config_path in config_paths]
        return cls(datasets)

    def save(self, save_path, store_rel_path=True):
        if os.path.isdir(save_path):
            save_config_path = os.path.join(save_path, 'config.json')
        else:
            save_config_path = save_path
            save_path = os.path.dirname(save_config_path)
        config_paths = [dataset.config_path for dataset in self.datasets]
        config_paths = sort_paths(config_paths)
        if store_rel_path:
            config_paths = [get_rel_path(config_path, save_path) for config_path in config_paths]
        config = {
            '_class_name': 'Dataset',
            'config_paths': config_paths,
        }
        utils.save_file(save_config_path, config)

    def open(self):
        for dataset in self.datasets:
            dataset.open()

    def close(self):
        for dataset in self.datasets:
            dataset.close()
        super(Dataset, self).close()

    def reset(self):
        for dataset in self.datasets:
            dataset.reset()
        super(Dataset, self).reset()

    def filter(self, mode, **kwargs):
        self.datasets[0].filter(mode, **kwargs)

    def __len__(self):
        return len(self.datasets[0])

    def _get_data(self, index):
        data_dict = self.datasets[0][index]
        if 'data_index' in data_dict:
            data_index = data_dict['data_index']
        else:
            data_index = index
        for i in range(1, len(self.datasets)):
            data_dict.update(self.datasets[i][data_index])
        return data_dict


class ConcatDataset(BaseDataset):
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0
        self.datasets = datasets

    @classmethod
    def load(cls, data_or_config_list):
        datasets = [load_dataset(d) for d in data_or_config_list]
        return cls(datasets)

    def open(self):
        for dataset in self.datasets:
            dataset.open()

    def close(self):
        for dataset in self.datasets:
            dataset.close()
        super(ConcatDataset, self).close()

    def reset(self):
        for dataset in self.datasets:
            dataset.reset()
        super(ConcatDataset, self).reset()

    def filter(self, mode, **kwargs):
        if mode.startswith('overall'):
            all_data_list = []
            for i, dataset in enumerate(self.datasets):
                dataset.datasets[0].open()
                all_data_list.append(dataset.datasets[0].data_list)
            if mode == 'overall_func':
                func = kwargs.pop('func')
                if isinstance(func, str):
                    func = utils.import_function(func)
                all_data_list = func(all_data_list, **kwargs)
            else:
                assert False
            for i, dataset in enumerate(self.datasets):
                logging.info(f'filter dataset {i} from {dataset.datasets[0].data_size} to {len(all_data_list[i])}')
                dataset.datasets[0].data_list = all_data_list[i]
                dataset.datasets[0].data_size = len(all_data_list[i])
        else:
            for i, dataset in enumerate(self.datasets):
                dataset.filter(mode=mode, dataset_index=i, **kwargs)

    def __len__(self):
        data_size = 0
        for dataset in self.datasets:
            data_size += len(dataset)
        return data_size

    def _get_data(self, index):
        for i, dataset in enumerate(self.datasets):
            if index >= len(dataset):
                index -= len(dataset)
            else:
                return dataset[index]
        raise IndexError
