import os

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin
from diffusers.models.attention_processor import (
    CROSS_ATTENTION_PROCESSORS,
    AttentionProcessor,
    AttnProcessor,
    IPAdapterAttnProcessor2_0,
)
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, GLIGENTextBoundingboxProjection
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_3d_blocks import UNetMidBlockSpatioTemporal, get_down_block, get_up_block
from diffusers.utils import BaseOutput, logging, WEIGHTS_NAME
from einops import rearrange

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class UNetSpatioTemporalConditionOutput(BaseOutput):
    """
    The output of [`UNetSpatioTemporalConditionModel`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            The hidden states output conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: torch.FloatTensor = None


class UNetSpatioTemporalConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""A conditional Spatio-Temporal UNet model that takes a noisy video
    frames, conditional state, and a timestep and returns a sample shaped
    output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 8): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "DownBlockSpatioTemporal")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        addition_time_embed_dim: (`int`, defaults to 256):
            Dimension to to encode the additional time ids.
        projection_class_embeddings_input_dim (`int`, defaults to 768):
            The dimension of the projection of encoded `added_time_ids`.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int`, `Tuple[int]`, or `Tuple[Tuple]` , *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unet_3d_blocks.CrossAttnDownBlockSpatioTemporal`],
            [`~models.unet_3d_blocks.CrossAttnUpBlockSpatioTemporal`],
            [`~models.unet_3d_blocks.UNetMidBlockSpatioTemporal`].
        num_attention_heads (`int`, `Tuple[int]`, defaults to `(5, 10, 10, 20)`):
            The number of attention heads.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        patch_size: int = 1,
        down_block_types: Tuple[str] = (
            'CrossAttnDownBlockSpatioTemporal',
            'CrossAttnDownBlockSpatioTemporal',
            'CrossAttnDownBlockSpatioTemporal',
            'DownBlockSpatioTemporal',
        ),
        up_block_types: Tuple[str] = (
            'UpBlockSpatioTemporal',
            'CrossAttnUpBlockSpatioTemporal',
            'CrossAttnUpBlockSpatioTemporal',
            'CrossAttnUpBlockSpatioTemporal',
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 10, 20),
        num_frames: int = 25,
        caption_channels: int = None,
        extra_caption: bool = False,
        num_tokens: int = 1,
        exvideo_cfg: dict = None,
        attention_type: str = 'default',
        add_emb: bool = False,
    ):
        super().__init__()

        self.sample_size = sample_size

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f'Must provide the same number of `down_block_types` as `up_block_types`. '
                f'`down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}.'
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f'Must provide the same number of `block_out_channels` as `down_block_types`. '
                f'`block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}.'
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f'Must provide the same number of `num_attention_heads` as `down_block_types`. '
                f'`num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}.'
            )

        if isinstance(cross_attention_dim, list) and len(cross_attention_dim) != len(down_block_types):
            raise ValueError(
                f'Must provide the same number of `cross_attention_dim` as `down_block_types`. '
                f'`cross_attention_dim`: {cross_attention_dim}. `down_block_types`: {down_block_types}.'
            )

        if not isinstance(layers_per_block, int) and len(layers_per_block) != len(down_block_types):
            raise ValueError(
                f'Must provide the same number of `layers_per_block` as `down_block_types`. '
                f'`layers_per_block`: {layers_per_block}. `down_block_types`: {down_block_types}.'
            )

        # input
        self.patch_size = patch_size
        if patch_size > 1:
            self.conv_in = nn.Conv2d(
                in_channels,
                block_out_channels[0],
                kernel_size=patch_size,
                stride=patch_size,
            )
        else:
            self.conv_in = nn.Conv2d(
                in_channels,
                block_out_channels[0],
                kernel_size=3,
                padding=1,
            )

        # time
        time_embed_dim = block_out_channels[0] * 4

        self.time_proj = Timesteps(block_out_channels[0], True, downscale_freq_shift=0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        if add_emb:
            self.add_time_proj = Timesteps(addition_time_embed_dim, True, downscale_freq_shift=0)
            self.add_embedding = TimestepEmbedding(projection_class_embeddings_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if isinstance(cross_attention_dim, int):
            cross_attention_dim = (cross_attention_dim,) * len(down_block_types)

        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)

        if isinstance(transformer_layers_per_block, int):
            transformer_layers_per_block = [transformer_layers_per_block] * len(down_block_types)

        blocks_time_embed_dim = time_embed_dim

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block[i],
                transformer_layers_per_block=transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=blocks_time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-5,
                cross_attention_dim=cross_attention_dim[i],
                num_attention_heads=num_attention_heads[i],
                resnet_act_fn='silu',
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockSpatioTemporal(
            block_out_channels[-1],
            temb_channels=blocks_time_embed_dim,
            transformer_layers_per_block=transformer_layers_per_block[-1],
            cross_attention_dim=cross_attention_dim[-1],
            num_attention_heads=num_attention_heads[-1],
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))
        reversed_layers_per_block = list(reversed(layers_per_block))
        reversed_cross_attention_dim = list(reversed(cross_attention_dim))
        reversed_transformer_layers_per_block = list(reversed(transformer_layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=reversed_layers_per_block[i] + 1,
                transformer_layers_per_block=reversed_transformer_layers_per_block[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=blocks_time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-5,
                resolution_idx=i,
                cross_attention_dim=reversed_cross_attention_dim[i],
                num_attention_heads=reversed_num_attention_heads[i],
                resnet_act_fn='silu',
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-5)
        self.conv_act = nn.SiLU()

        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels * patch_size * patch_size,
            kernel_size=3,
            padding=1,
        )

        self.num_frames = num_frames
        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=caption_channels, hidden_size=cross_attention_dim[0]
            )
        if extra_caption:
            self.set_attn_processors_for_caption(num_tokens)
        if exvideo_cfg is not None:
            self.enable_exvideo(**exvideo_cfg)
    
        if attention_type in ['gated', 'gated-text-image']:
            positive_len = 768
            if isinstance(cross_attention_dim, int):
                positive_len = cross_attention_dim
            elif isinstance(cross_attention_dim, tuple) or isinstance(cross_attention_dim, list):
                positive_len = cross_attention_dim[0]

            feature_type = 'text-only' if attention_type == 'gated' else 'text-image'
            self.position_net = GLIGENTextBoundingboxProjection(
                positive_len=positive_len, out_dim=cross_attention_dim, feature_type=feature_type
            )


    @classmethod
    def from_unet(cls, unet, dtype_for_2d, dtype_for_3d=torch.float32, tune_all_unet_params=False, **kwargs):
        in_channels=unet.in_channels
        cross_attention_dim=unet.cross_attention_dim
        num_attention_heads=unet.attention_head_dim
        sample_size=unet.sample_size  # no meaning, for compatibility is_unet_version_less_0_9_0 in diffusers
        unet3d_model_path = kwargs.pop('unet3d_model_path', None)
        unet3d = cls(
            in_channels=in_channels,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads,
            sample_size=sample_size,
            **kwargs,
        )
        
        loaded_names = []
        missing_names = []
        for name, param in unet3d.named_parameters():
            if name in unet.state_dict():
                tmp_data = unet.state_dict()[name]
                param.data = tmp_data.reshape(param.data.shape)
                if tune_all_unet_params:
                    param.requires_grad_(True)
                    param.data = param.to(dtype_for_3d)
                else:
                    param.requires_grad_(False)
                    param.data = param.to(dtype_for_2d)
                loaded_names.append(name)
            elif 'spatial_res_block.' in name:
                name_in_2d = name.replace('spatial_res_block.', '')
                if name_in_2d in unet.state_dict():
                    param.data = unet.state_dict()[name_in_2d]
                    if tune_all_unet_params:
                        param.requires_grad_(True)
                        param.data = param.to(dtype_for_3d)
                    else:
                        param.requires_grad_(False)
                        param.data = param.to(dtype_for_2d)
                    loaded_names.append(name)
                else:   
                    param.requires_grad_(True)
                    param.to(dtype_for_3d)
                    missing_names.append(name)
            else:
                missing_names.append(name)
                param.to(dtype_for_3d)
                param.requires_grad_(True)
        assert len(loaded_names) == len(unet.state_dict())
        if unet3d_model_path is not None:
            unet_3d_weights = torch.load(os.path.join(unet3d_model_path, WEIGHTS_NAME))
            for name, param in unet3d.named_parameters():
                if name in unet_3d_weights:
                    tmp_data = unet_3d_weights[name]
                    param.data = tmp_data.reshape(param.data.shape)
                    if tune_all_unet_params:
                        param.requires_grad_(True)
                        param.data = param.to(dtype_for_3d)
                    else:
                        param.requires_grad_(False)
                        param.data = param.to(dtype_for_2d)
                    if name in missing_names:
                        missing_names.remove(name)
            assert len(missing_names) == 0
        return unet3d
    def set_num_frames(self, num_frames):
        self.num_frames = num_frames

    def enable_exvideo(self, *args, **kwargs):
        for name, module in self.named_modules():
            if hasattr(module, 'train_timesteps'):
                module.train_timesteps(*args, **kwargs)

    def set_attn_processors_for_caption(self, num_tokens):
        for name, module in self.named_modules():
            if module.__class__.__name__ == 'Attention':
                if '.attn2' in name:
                    to_k_shape = module.to_k.weight.shape
                    processor = IPAdapterAttnProcessor2_0(
                        hidden_size=to_k_shape[0],
                        cross_attention_dim=to_k_shape[1],
                        num_tokens=num_tokens,
                    )
                    module.set_processor(processor)

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, 'get_processor'):
                processors[f'{name}.processor'] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f'{name}.{sub_name}', child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.
        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f'A dict of processors was passed, but the number of processors {len(processor)} does not match the'
                f' number of attention layers: {count}. Please make sure to pass {count} processor classes.'
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, 'set_processor'):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f'{name}.processor'))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f'{name}.{sub_name}', child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """Disables custom attention processors and sets the default attention
        implementation."""
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f'Cannot call `set_default_attn_processor` when attention processors are of type '
                f'{next(iter(self.attn_processors.values()))}'
            )

        self.set_attn_processor(processor)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f'Make sure to set `dim` to either 0 or 1, not {dim}')

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, 'set_chunk_feed_forward'):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        conv_in_additional_residual: Optional[torch.Tensor] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        added_time_ids: torch.Tensor = None,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        r"""
        The [`UNetSpatioTemporalConditionModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            encoder_hidden_states (`torch.FloatTensor`):
                The encoder hidden states with shape `(batch, sequence_length, cross_attention_dim)`.
            added_time_ids: (`torch.FloatTensor`):
                The additional time ids with shape `(batch, num_additional_ids)`. These are encoded with sinusoidal
                embeddings and added to the time embeddings.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor]
                (https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`]
                instead of a plain tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`]
                is returned, otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == 'mps'
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size = sample.shape[0] // self.num_frames

        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        if added_time_ids is not None:
            time_embeds = self.add_time_proj(added_time_ids.flatten())
            time_embeds = time_embeds.reshape((added_time_ids.shape[0], -1))
            time_embeds = time_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(time_embeds)
            emb = emb + aug_emb

        # 2. pre-process
        sample = self.conv_in(sample)

        if conv_in_additional_residual is not None:
            sample = sample + conv_in_additional_residual

        # 2.5 GLIGEN position net
        if cross_attention_kwargs is not None and cross_attention_kwargs.get('gligen', None) is not None:
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop('gligen')
            cross_attention_kwargs['gligen'] = {'objs': self.position_net(**gligen_args)}
            
        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        image_only_indicator = torch.zeros(batch_size, self.num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            cross_attention_kwargs=cross_attention_kwargs,
            image_only_indicator=image_only_indicator,
        )

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if self.patch_size > 1:
            sample = rearrange(sample, 'b (p q c) h w -> b c (h p) (w q)', p=self.patch_size, q=self.patch_size)

        if not return_dict:
            return (sample,)

        return UNetSpatioTemporalConditionOutput(sample=sample)

