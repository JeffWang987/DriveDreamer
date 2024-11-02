import functools

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import GatedSelfAttentionDense
from diffusers.models.modeling_utils import ModelMixin

from .grounding_downsampler import GroundingDownSampler
from .position_net import ImagePositionNet, PositionNet


class DriveDreamerModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        conv_in_cfg=None,
        grounding_downsampler_cfg=None,
        position_net_cfg=None,
        fusers_cfg=None,
    ):
        super().__init__()
        if conv_in_cfg is not None:
            conv_in_cfg = conv_in_cfg.copy()
            conv_in_type = conv_in_cfg.pop('type')
            if conv_in_type == 'Conv2d':
                self.conv_in = nn.Conv2d(**conv_in_cfg)
            else:
                assert False
        else:
            self.conv_in = None
        if grounding_downsampler_cfg is not None:
            grounding_downsampler_cfg = grounding_downsampler_cfg.copy()
            gd_type = grounding_downsampler_cfg.pop('type')
            if gd_type == 'GroundingDownSampler':
                self.grounding_downsampler = GroundingDownSampler(**grounding_downsampler_cfg)
            else:
                assert False
        else:
            self.grounding_downsampler = None
        if position_net_cfg is not None:
            position_net_cfg = position_net_cfg.copy()
            pos_type = position_net_cfg.pop('type')
            if pos_type == 'PositionNet':
                self.position_net = PositionNet(**position_net_cfg)
            elif pos_type == 'ImagePositionNet':
                self.position_net = ImagePositionNet(**position_net_cfg)
            else:
                assert False
        else:
            self.position_net = None
        if fusers_cfg is not None:
            self.fuser_names = []
            self.fusers = nn.ModuleList()
            for fuser_name, fuser_cfg in fusers_cfg:
                fuser_cfg = fuser_cfg.copy()
                fuser_type = fuser_cfg.pop('type')
                if fuser_type == 'GatedSelfAttentionDense':
                    fuser = GatedSelfAttentionDense(**fuser_cfg)
                else:
                    assert False
                self.fuser_names.append(fuser_name)
                self.fusers.append(fuser)
        else:
            self.fuser_names = []
            self.fusers = None
        self.gradient_checkpointing = False

    @classmethod
    def from_unet(cls, unet, add_in_channels=0, **kwargs):
        if add_in_channels != 0:
            conv_in_kernel = unet.config.get('conv_in_kernel', 3)
            conv_in_padding = (conv_in_kernel - 1) // 2
            conv_in_cfg = dict(
                type='Conv2d',
                in_channels=unet.config.in_channels + add_in_channels,
                out_channels=unet.config.block_out_channels[0],
                kernel_size=conv_in_kernel,
                padding=conv_in_padding,
            )
        else:
            conv_in_cfg = None
        fusers_cfg = []
        for name, module in unet.named_modules():
            if module.__class__.__name__ == 'BasicTransformerBlock' and '.attentions.' in name:
                query_dim = module.attn1.to_out[0].weight.shape[0]
                cross_attention_dim = unet.config.cross_attention_dim
                heads = module.attn1.heads
                dim_head = module.attn1.inner_dim // heads
                fuser_cfg = dict(
                    type='GatedSelfAttentionDense',
                    query_dim=query_dim,
                    context_dim=cross_attention_dim,
                    n_heads=heads,
                    d_head=dim_head,
                )
                fusers_cfg.append((name, fuser_cfg))
        if 'position_net_cfg' in kwargs:
            kwargs['position_net_cfg']['out_dim'] = unet.config.cross_attention_dim
        drivedreamer = cls(
            conv_in_cfg=conv_in_cfg,
            fusers_cfg=fusers_cfg,
            **kwargs,
        )
        return drivedreamer

    def to_unet(self, unet):
        if self.conv_in is not None:
            del unet.conv_in
            unet.conv_in = functools.partial(self, 'conv_in')
        if self.fusers is not None:
            for name, module in unet.named_modules():
                if module.__class__.__name__ == 'BasicTransformerBlock' and '.attentions.' in name:
                    module.fuser = functools.partial(self, name)
        unet.position_net = Identity()

    def load_state_dict(self, state_dict, *args, **kwargs):
        if 'conv_in.weight' in state_dict:
            conv_in_weight = state_dict['conv_in.weight']
            if conv_in_weight.numel() > 0 and conv_in_weight.shape[1] != self.conv_in.weight.shape[1]:
                assert self.conv_in.weight.shape[1] >= conv_in_weight.shape[1]
                new_conv_in_weight = conv_in_weight.new_zeros(self.conv_in.weight.shape)
                new_conv_in_weight[:, : conv_in_weight.shape[1]] = conv_in_weight
                state_dict['conv_in.weight'] = new_conv_in_weight
        return super().load_state_dict(state_dict, *args, **kwargs)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(self, key, *args, **kwargs):
        if key in ('conv_in', 'grounding_downsampler', 'position_net'):
            module = getattr(self, key)
        elif key in self.fuser_names:
            idx = self.fuser_names.index(key)
            module = self.fusers[idx]
        else:
            assert False
        if self.training and self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(module, *args, use_reentrant=False, **kwargs)
        else:
            return module(*args, **kwargs)


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, objs):
        return objs
