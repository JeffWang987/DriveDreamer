import functools

import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.lora import LoRALinearLayer
from diffusers.models.modeling_utils import ModelMixin


class LoRAModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, lora_layers_cfg):
        super().__init__()
        self.model_names = []
        self.lora_layer_names = []
        self.lora_layers = nn.ModuleList()
        for lora_layer_name, lora_layer_cfg in lora_layers_cfg:
            lora_layer = nn.ModuleDict()
            for l_name, l_cfg in lora_layer_cfg.items():
                l_cfg = l_cfg.copy()
                l_type = l_cfg.pop('type')
                if l_type == 'LoRALinearLayer':
                    lora_layer[l_name] = LoRALinearLayer(**l_cfg)
                else:
                    assert False
            assert lora_layer_name not in self.lora_layer_names
            model_name = lora_layer_name.split('.')[0]
            if model_name not in self.model_names:
                self.model_names.append(model_name)
            self.lora_layer_names.append(lora_layer_name)
            self.lora_layers.append(lora_layer)

    @classmethod
    def from_models(cls, models, rank, conditions=None):
        lora_layers_cfg = []
        for i, model in enumerate(models):
            model_name = model.__class__.__name__
            for module_name, module in model.named_modules():
                if module.__class__.__name__ == 'Attention':
                    name = f'{model_name}.{module_name}'
                    if conditions is not None and not meet_condition(name, conditions[i]):
                        continue
                    lora_layer_cfg = dict(
                        to_q=dict(
                            type='LoRALinearLayer',
                            in_features=module.to_q.in_features,
                            out_features=module.to_q.out_features,
                            rank=rank,
                        ),
                        to_k=dict(
                            type='LoRALinearLayer',
                            in_features=module.to_k.in_features,
                            out_features=module.to_k.out_features,
                            rank=rank,
                        ),
                        to_v=dict(
                            type='LoRALinearLayer',
                            in_features=module.to_v.in_features,
                            out_features=module.to_v.out_features,
                            rank=rank,
                        ),
                        to_out=dict(
                            type='LoRALinearLayer',
                            in_features=module.to_out[0].in_features,
                            out_features=module.to_out[0].out_features,
                            rank=rank,
                        ),
                    )
                    lora_layers_cfg.append((name, lora_layer_cfg))
        lora = cls(lora_layers_cfg=lora_layers_cfg)
        return lora

    def _fuse_lora(self, module, key, key2, scale=1.0, *args, **kwargs):
        idx = self.lora_layer_names.index(key)
        lora_layer = self.lora_layers[idx][key2]
        if scale != 1.0:
            lora_layer.network_alpha = scale * lora_layer.rank
        module.lora_layer = lora_layer
        module._fuse_lora(*args, **kwargs)

    def to_models(self, models, scale=1.0, is_train=True):
        for model in models:
            model_name = model.__class__.__name__
            if model_name not in self.model_names:
                continue
            print(f'Load lora of {model_name}')
            for module_name, module in model.named_modules():
                if module.__class__.__name__ == 'Attention':
                    name = f'{model_name}.{module_name}'
                    if name not in self.lora_layer_names:
                        continue
                    if is_train:
                        assert scale == 1.0
                        module.to_q.lora_layer = functools.partial(self, name, 'to_q')
                        module.to_k.lora_layer = functools.partial(self, name, 'to_k')
                        module.to_v.lora_layer = functools.partial(self, name, 'to_v')
                        module.to_out[0].lora_layer = functools.partial(self, name, 'to_out')
                    else:
                        self._fuse_lora(module.to_q, name, 'to_q', scale)
                        self._fuse_lora(module.to_k, name, 'to_k', scale)
                        self._fuse_lora(module.to_v, name, 'to_v', scale)
                        self._fuse_lora(module.to_out[0], name, 'to_out', scale)

    def forward(self, key, key2, *args, **kwargs):
        idx = self.lora_layer_names.index(key)
        module = self.lora_layers[idx][key2]
        return module(*args, **kwargs)


def meet_condition(name, conditions):
    if conditions is None:
        return True
    for condition in conditions:
        if condition not in name:
            return False
    return True
