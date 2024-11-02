import copy
import json
import os
import pickle
import pprint

import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader

from ..utils import import_function


def load_file(file_path, **kwargs):
    if file_path.endswith('.pkl') or file_path.endswith('.pickle'):
        data = pickle.load(open(file_path, 'rb'), **kwargs)
    elif file_path.endswith('.json'):
        data = json.load(open(file_path, 'r'), **kwargs)
    elif file_path.endswith('.yaml') or file_path.endswith('yml'):
        kwargs.setdefault('Loader', Loader)
        data = yaml.load(open(file_path, 'r'), **kwargs)
    else:
        assert False
    return data


def save_file(file_path, data, **kwargs):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if file_path.endswith('.pkl') or file_path.endswith('.pickle'):
        pickle.dump(data, open(file_path, 'wb'), **kwargs)
    elif file_path.endswith('.json'):
        kwargs.setdefault('indent', 4)
        json.dump(data, open(file_path, 'w'), **kwargs)
    elif file_path.endswith('.yaml') or file_path.endswith('yml'):
        kwargs.setdefault('Dumper', Dumper)
        yaml.dump(data, open(file_path, 'w'), **kwargs)
    else:
        assert False


def load_config(config_or_path):
    if isinstance(config_or_path, str):
        if os.path.isdir(config_or_path):
            config_path = os.path.join(config_or_path, 'config.json')
        else:
            config_path = config_or_path
        config = Config.load(config_path)
    elif isinstance(config_or_path, Config):
        config = config_or_path
    elif isinstance(config_or_path, dict):
        config = Config(config_or_path)
    else:
        assert False
    return config


class Config(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)

    def _process_value(self, value):
        assert value is None or isinstance(value, (int, float, bool, str, list, tuple, dict, self.__class__))
        if isinstance(value, (list, tuple)):
            if len(value) > 0 and value[0] == '__tuple__':
                value = tuple(value[1:])
            is_tuple = isinstance(value, tuple)
            value = [self._process_value(x) if isinstance(x, (list, tuple, dict)) else x for x in value]
            if is_tuple:
                value = tuple(value)
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        return value

    def __setattr__(self, name, value):
        value = self._process_value(value)
        super(Config, self).__setattr__(name, value)
        super(Config, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __str__(self):
        return 'Config:\n{}'.format(self.pretty_text)

    @property
    def pretty_text(self):
        return pprint.pformat(self)

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k, v in d.items():
            if hasattr(self, k):
                force = v.pop('__force__', False) if isinstance(v, dict) else False
                if isinstance(v, dict) and isinstance(self[k], dict) and not force:
                    self[k].update(v)
                else:
                    setattr(self, k, v)
            else:
                setattr(self, k, v)
        return self

    def pop(self, k, d=None):
        if hasattr(self, k):
            delattr(self, k)
        return super(Config, self).pop(k, d)

    def setdefault(self, k, d=None):
        if hasattr(self, k):
            return getattr(self, k)
        else:
            setattr(self, k, d)
            return d

    def copy(self):
        return copy.deepcopy(self)

    def _to_dict(self, data, tuple_as_list=False):
        if isinstance(data, (dict, self.__class__)):
            return {k: self._to_dict(d, tuple_as_list) for k, d in data.items()}
        elif isinstance(data, (list, tuple)):
            new_data = [self._to_dict(d, tuple_as_list) for d in data]
            if isinstance(data, tuple):
                if tuple_as_list:
                    new_data = ['__tuple__'] + new_data
                else:
                    new_data = tuple(new_data)
            return new_data
        else:
            return data

    def to_dict(self, tuple_as_list=False):
        return self._to_dict(self, tuple_as_list=tuple_as_list)

    def save(self, filename):
        tuple_as_list = True if filename.endswith('.json') else False
        data = self.to_dict(tuple_as_list=tuple_as_list)
        save_file(filename, data)

    @classmethod
    def load(cls, filename):
        if filename.endswith('.config'):
            config = import_function(filename)
        elif filename.endswith('.py'):
            config = import_function(filename[:-3] + '/config', '/')
        else:
            config = load_file(filename)
        return cls(config)

    def merge_from_file(self, filename):
        self.update(Config.load(filename))
        return self
