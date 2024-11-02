from tqdm import tqdm
from dreamer_models import utils as gm_utils


def save_ema(ckpt_paths, save_path, gamma=0.9):
    state_dict_ema = None
    dtype = None
    for i in tqdm(range(len(ckpt_paths))):
        state_dict = gm_utils.load_state_dict(ckpt_paths[i])
        if i == 0:
            dtype = list(state_dict.values())[0].dtype
            state_dict_ema = {name: param.float() for name, param in state_dict.items()}
        else:
            for name, param in state_dict.items():
                state_dict_ema[name] = state_dict_ema[name] * gamma + param.float() * (1 - gamma)
    state_dict_ema = {name: param.to(dtype) for name, param in state_dict_ema.items()}
    gm_utils.save_state_dict(state_dict_ema, save_path)
