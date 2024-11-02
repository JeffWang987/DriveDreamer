import os

import safetensors
import torch
from diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME


def wrap_call(func):
    def f(*args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(e)
                continue

    return f


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(int(os.environ.get('GIGA_MODELS_DEFAULT_DEVICE', '0'))))
    else:
        device = torch.device('cpu')
    return device


def load_state_dict(weight_path):
    if os.path.isdir(weight_path):
        if os.path.exists(os.path.join(weight_path, WEIGHTS_NAME)):
            return torch.load(os.path.join(weight_path, WEIGHTS_NAME), map_location='cpu')
        elif os.path.exists(os.path.join(weight_path, SAFETENSORS_WEIGHTS_NAME)):
            return safetensors.torch.load_file(os.path.join(weight_path, SAFETENSORS_WEIGHTS_NAME), device='cpu')
        else:
            assert False
    elif os.path.isfile(weight_path):
        if weight_path.endswith('.safetensors'):
            return safetensors.torch.load_file(weight_path, device='cpu')
        else:
            return torch.load(weight_path, map_location='cpu')
    else:
        assert False


def save_state_dict(state_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.endswith('.safetensors'):
        safetensors.torch.save_file(state_dict, save_path)
    else:
        torch.save(state_dict, save_path)
