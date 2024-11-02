import json
import os
import pickle
import shutil
import stat

import yaml

try:
    from yaml import CDumper as Dumper
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Dumper, Loader


def empty_dir(root_dir):
    if os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)


def list_dir(root_dir, recursive=False, exts=None):
    file_paths = []
    if recursive:
        for cur_dir, _, file_names in os.walk(root_dir, followlinks=True):
            for file_name in file_names:
                file_path = os.path.join(cur_dir, file_name)
                if os.path.isfile(file_path):
                    if exts is None:
                        file_paths.append(file_path)
                    else:
                        suffix = os.path.splitext(file_name)[1].lower()
                        if suffix in exts:
                            file_paths.append(file_path)
    else:
        for file_name in sorted(os.listdir(root_dir)):
            file_path = os.path.join(root_dir, file_name)
            if os.path.isfile(file_path):
                if exts is None:
                    file_paths.append(file_path)
                else:
                    suffix = os.path.splitext(file_name)[1].lower()
                    if suffix in exts:
                        file_paths.append(file_path)
    file_paths.sort()
    return file_paths


def chmod_dir(root_dir, mode=stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO):
    for cur_dir, _, file_names in os.walk(root_dir, followlinks=True):
        os.chmod(cur_dir, mode)
        for file_name in file_names:
            file_path = os.path.join(cur_dir, file_name)
            if os.path.isfile(file_path):
                os.chmod(file_path, mode)


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
