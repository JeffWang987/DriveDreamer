import numpy as np
import torch


def slice_data(data, slc, dim=0):
    if isinstance(data, (np.ndarray, torch.Tensor)):
        if data.ndim > 0 and data.shape[dim] > max(slc):
            if dim == 0:
                return data[slc]
            elif dim == 1:
                return data[:, slc]
            elif dim == 2:
                return data[:, :, slc]
            elif dim == 3:
                return data[:, :, :, slc]
            elif dim == 4:
                return data[:, :, :, :, slc]
            elif dim == 5:
                return data[:, :, :, :, :, slc]
            else:
                assert False
        else:
            return data
    elif isinstance(data, (list, tuple)):
        return type(data)([slice_data(d, slc, dim) for d in data])
    elif isinstance(data, dict):
        return {key: slice_data(data[key], slc, dim) for key in data}
    else:
        return data


def merge_data(data_list, slices, dim=0):
    if isinstance(data_list[0], (np.ndarray, torch.Tensor)):
        max_size = int(max(max(slc) for slc in slices)) + 1
        data_shape = list(data_list[0].shape)
        data_shape[dim] = max_size
        if isinstance(data_list[0], np.ndarray):
            new_data = np.zeros(data_shape, dtype=data_list[0].dtype)
            count = np.zeros((max_size,), dtype=data_list[0].dtype)
        else:
            new_data = torch.zeros(data_shape, dtype=data_list[0].dtype, device=data_list[0].device)
            count = torch.zeros((max_size,), dtype=data_list[0].dtype, device=data_list[0].device)
        for data, slc in zip(data_list, slices):
            if dim == 0:
                new_data[slc] = new_data[slc] + data
            elif dim == 1:
                new_data[:, slc] = new_data[:, slc] + data
            elif dim == 2:
                new_data[:, :, slc] = new_data[:, :, slc] + data
            elif dim == 3:
                new_data[:, :, :, slc] = new_data[:, :, :, slc] + data
            elif dim == 4:
                new_data[:, :, :, :, slc] = new_data[:, :, :, :, slc] + data
            elif dim == 5:
                new_data[:, :, :, :, :, slc] = new_data[:, :, :, :, :, slc] + data
            else:
                assert False
            count[slc] += 1
        count_shape = [1 for _ in data_shape]
        count_shape[dim] = max_size
        count = count.reshape(count_shape)
        new_data = new_data / count
        return new_data
    elif isinstance(data_list[0], (list, tuple)):
        return type(data_list[0])(merge_data(d, slices, dim) for d in zip(*data_list))
    else:
        return data_list


def apply_slice(func, *args, slices, dim=0, **kwargs):
    outputs = []
    for slc in slices:
        args_i = slice_data(args, slc, dim=dim)
        kwargs_i = slice_data(kwargs, slc, dim=dim)
        outputs_i = func(*args_i, **kwargs_i)
        outputs.append(outputs_i)
    outputs = merge_data(outputs, slices, dim=dim)
    return outputs


def apply_chunk(func, *args, chunk_size=1, dim=0, **kwargs):
    data_size = None
    for i in range(len(args)):
        if data_size is None:
            data_size = args[i].shape[dim]
        else:
            assert data_size == args[i].shape[dim]
    for key in kwargs:
        if data_size is None:
            data_size = kwargs[key].shape[dim]
        else:
            assert data_size == kwargs[key].shape[dim]
    assert data_size is not None
    num_chunks = (data_size - 1) // chunk_size + 1
    slices = torch.arange(data_size, dtype=torch.int64)
    slices = slices.chunk(num_chunks, dim=0)
    return apply_slice(func, *args, slices=slices, dim=dim, **kwargs)
