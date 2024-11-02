import time
from importlib import import_module

import pynvml


def as_list(data):
    if isinstance(data, list):
        return data
    elif isinstance(data, tuple):
        return list(data)
    else:
        return [data]


def import_function(function_name, sep='.'):
    parts = function_name.split(sep)
    module_name = '.'.join(parts[:-1])
    module = import_module(module_name)
    return getattr(module, parts[-1])


def get_cur_time():
    return time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))


def wait_for_gpu_memory(gpu_ids, gpu_memory, unit='GB', seconds=10, count_limit=-1):
    factors = {
        'GB': 1024 * 1024 * 1024,
        'MB': 1024 * 1024,
        'KB': 1024,
        'B': 1,
    }
    factor = factors[unit]
    pynvml.nvmlInit()
    gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(gpu_id) for gpu_id in gpu_ids]
    count = 0
    while True:
        meet_need = True
        for gpu_id, gpu_handle in zip(gpu_ids, gpu_handles):
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            mem_total = mem_info.total / factor
            mem_used = mem_info.used / factor
            mem_free = mem_info.free / factor
            if mem_free < gpu_memory:
                meet_need = False
            msg = 'GPU {}:'.format(gpu_id)
            msg += ' Total {:.2f}{}'.format(mem_total, unit)
            msg += ', Used {:.2f}{}'.format(mem_used, unit)
            msg += ', Free {:.2f}{}'.format(mem_free, unit)
            print(msg)
        if meet_need:
            break
        else:
            count += 1
            if count > count_limit > 0:
                print('Timeout Exit')
                exit(0)
            else:
                print('Wait For GPU Memory: {:.2f}{} .....'.format(gpu_memory, unit))
                time.sleep(seconds)
