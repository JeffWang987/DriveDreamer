import os


def get_root_dir():
    return os.path.abspath(__file__).split('dreamer_models')[0][:-1]


def get_model_dir():
    return os.environ.get('DREAMER_MODELS_DIR', './models/')


def get_huggingface_model_path(model_name):
    model_dir = os.environ['HUGGINGFACE_HUB_CACHE']
    if '/' in model_name and len(model_name.split('/')) == 2:
        local_model_name = 'models--' + model_name.replace('/', '--')
        hf_model_name = model_name
    elif '--' in model_name and model_name.startswith('models'):
        local_model_name = model_name
        hf_model_name = model_name[8:]
        hf_model_name = hf_model_name.replace('--', '/')
    else:
        local_model_name = model_name
        hf_model_name = None
    model_path = os.path.join(model_dir, local_model_name)
    if not os.path.exists(model_path):
        raise ValueError(f'{model_path} does not exist')
    
    # if os.path.exists(os.path.join(model_path, 'refs')):
    #     return hf_model_name
    # else:
    #     return model_path
    return model_path


def get_model_path(model_name_or_path):
    if model_name_or_path is None or os.path.exists(model_name_or_path):
        return model_name_or_path
    if os.path.isabs(model_name_or_path):
        raise ValueError(f'{model_name_or_path} does not exist')
    model_dir = get_model_dir()
    model_path = os.path.join(model_dir, model_name_or_path)
    if os.path.exists(model_path):
        return model_path
    return get_huggingface_model_path(model_name_or_path)
