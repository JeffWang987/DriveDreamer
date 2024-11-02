from importlib import import_module

import numpy as np
import torch


def as_list(data):
    if isinstance(data, list):
        return data
    else:
        return [data]


def repeat_n(data, n):
    if not isinstance(data, list):
        data = [data]
    if len(data) == 1:
        data = data * n
    else:
        assert len(data) == n
    return data


def stack_data(data_list, pad_value=0):
    equal_shape = True
    for data in data_list:
        if data.shape != data_list[0].shape:
            equal_shape = False
            break
    if equal_shape:
        if isinstance(data_list[0], np.ndarray):
            new_data = np.stack(data_list)
        elif isinstance(data_list[0], torch.Tensor):
            new_data = torch.stack(data_list)
        else:
            assert False
    else:
        ndim = data_list[0].ndim
        dtype = data_list[0].dtype
        new_data_shape = [len(data_list)]
        for dim in range(ndim):
            new_data_shape.append(max(data.shape[dim] for data in data_list))
        if isinstance(data_list[0], np.ndarray):
            new_data = np.full(tuple(new_data_shape), pad_value, dtype=dtype)
        elif isinstance(data_list[0], torch.Tensor):
            new_data = torch.full(tuple(new_data_shape), pad_value, dtype=dtype)
        else:
            assert False
        for i, data in enumerate(data_list):
            if ndim == 1:
                new_data[i][: data.shape[0]] = data
            elif ndim == 2:
                new_data[i][: data.shape[0], : data.shape[1]] = data
            elif ndim == 3:
                new_data[i][: data.shape[0], : data.shape[1], : data.shape[2]] = data
            elif ndim == 4:
                new_data[i][: data.shape[0], : data.shape[1], : data.shape[2], : data.shape[3]] = data
            else:
                assert False
    return new_data


def wrap_call(func):
    def f(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(e)
                continue

    return f


def import_function(function_name, sep='.'):
    parts = function_name.split(sep)
    module_name = '.'.join(parts[:-1])
    module = import_module(module_name)
    return getattr(module, parts[-1])
