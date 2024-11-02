from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.embeddings import TimestepEmbedding, Timesteps, get_timestep_embedding
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.resnet import AlphaBlender
from diffusers.models.transformers import transformer_temporal
from diffusers.models.transformers.transformer_temporal import TransformerTemporalModelOutput
from diffusers.models.unets import unet_3d_blocks
from einops import rearrange

from .attention import TemporalBasicTransformerBlock


class TransformerTemporalModel(ModelMixin, ConfigMixin):
    """A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlock` attention should contain a bias parameter.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward. See `diffusers.models.activations.get_activation` for supported
            activation functions.
        norm_elementwise_affine (`bool`, *optional*):
            Configure if the `TransformerBlock` should use learnable elementwise affine parameters for normalization.
        double_self_attention (`bool`, *optional*):
            Configure if each `TransformerBlock` should contain two self-attention layers.
        positional_embeddings: (`str`, *optional*):
            The type of positional embeddings to apply to the sequence input before passing use.
        num_positional_embeddings: (`int`, *optional*):
            The maximum length of the sequence over which to apply positional embeddings.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        activation_fn: str = 'geglu',
        norm_elementwise_affine: bool = True,
        double_self_attention: bool = True,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    double_self_attention=double_self_attention,
                    norm_elementwise_affine=norm_elementwise_affine,
                    positional_embeddings=positional_embeddings,
                    num_positional_embeddings=num_positional_embeddings,
                )
                for d in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(inner_dim, in_channels)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.LongTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: torch.LongTensor = None,
        num_frames: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> TransformerTemporalModelOutput:
        """
        The [`TransformerTemporal`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete,
                           `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input hidden_states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            num_frames (`int`, *optional*, defaults to 1):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames

        residual = hidden_states

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(batch_size * height * width, num_frames, channel)

        hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = (
            hidden_states[None, None, :]
            .reshape(batch_size, height, width, num_frames, channel)
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
        hidden_states = hidden_states.reshape(batch_frames, channel, height, width)

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)


class TransformerSpatioTemporalModel(nn.Module):
    """A Transformer model for video-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        out_channels (`int`, *optional*):
            The number of channels in the output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: int = 320,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        inner_dim = num_attention_heads * attention_head_dim
        self.inner_dim = inner_dim

        # 2. Define input layers
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for d in range(num_layers)
            ]
        )

        time_mix_inner_dim = inner_dim
        self.temporal_transformer_blocks = nn.ModuleList(
            [
                TemporalBasicTransformerBlock(
                    inner_dim,
                    time_mix_inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                )
                for _ in range(num_layers)
            ]
        )

        time_embed_dim = in_channels * 4
        self.time_pos_embed = TimestepEmbedding(in_channels, time_embed_dim, out_dim=in_channels)
        self.time_proj = Timesteps(in_channels, True, 0)
        self.time_conv = None
        self.time_mixer = AlphaBlender(alpha=0.5, merge_strategy='learned_with_images')

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        # TODO: should use out_channels for continuous projections
        self.proj_out = nn.Linear(inner_dim, in_channels)

        self.gradient_checkpointing = False

    def train_timesteps(self, num_frames, max_id=25, repeat_length=20):
        self.time_proj = nn.ModuleList(
            [
                TrainableTemporalTimesteps(max_id, repeat_length, self.in_channels, True, 0, num_frames)
                for d in range(len(self.temporal_transformer_blocks))
            ]
        )
        self.time_conv = nn.ModuleList(
            [
                torch.nn.Conv3d(self.in_channels, self.in_channels, kernel_size=3, padding=1, padding_mode='reflect')
                for d in range(len(self.temporal_transformer_blocks))
            ]
        )
        _init_time_conv(self.time_conv)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        image_only_indicator: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input hidden_states.
            num_frames (`int`):
                The number of frames to be processed per batch. This is used to reshape the hidden states.
            encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            image_only_indicator (`torch.LongTensor` of shape `(batch size, num_frames)`, *optional*):
                A tensor indicating whether the input contains only images. 1 indicates that the input contains only
                images, 0 indicates that the input contains video frames.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_temporal.TransformerTemporalModelOutput`]
                instead of a plain tuple.

        Returns:
            [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                returned, otherwise a `tuple` where the first element is the sample tensor.
        """
        # 1. Input
        batch_frames, _, height, width = hidden_states.shape
        num_frames = image_only_indicator.shape[-1]
        batch_size = batch_frames // num_frames

        time_context = encoder_hidden_states
        time_context_first_timestep = time_context.reshape(batch_size, num_frames, -1, time_context.shape[-1])[:, 0]
        time_context = time_context_first_timestep[None, :].broadcast_to(
            height * width, batch_size, time_context.shape[-2], time_context.shape[-1]
        )
        time_context = time_context.permute(1, 0, 2, 3)
        time_context = time_context.reshape(batch_size * height * width, time_context.shape[-2], time_context.shape[-1])

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        inner_dim = hidden_states.shape[1]
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
        hidden_states = self.proj_in(hidden_states)

        if self.time_conv is None:
            num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
            t_emb = self.time_proj(num_frames_emb)
            t_emb = t_emb.to(dtype=hidden_states.dtype)
            emb = self.time_pos_embed(t_emb)
            emb = emb[:, None, :]
            emb = emb.repeat(batch_size, 1, 1)

        # 2. Blocks
        for i, (block, temporal_block) in enumerate(zip(self.transformer_blocks, self.temporal_transformer_blocks)):
            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    block,
                    hidden_states,
                    None,
                    encoder_hidden_states,
                    None,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                )

            hidden_states_mix = hidden_states

            if self.time_conv is None:
                hidden_states_mix = hidden_states_mix + emb
            else:
                num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
                t_emb = self.time_proj[i](num_frames_emb)
                t_emb = t_emb.to(dtype=hidden_states.dtype)
                emb = self.time_pos_embed(t_emb)
                emb = emb[:, None, :]
                emb = emb.repeat(batch_size, 1, 1)
                hidden_states_mix = hidden_states_mix + emb
                hidden_states_mix = rearrange(
                    hidden_states_mix, '(B T) (H W) C -> B C T H W', B=batch_size, T=num_frames, H=height, W=width
                )
                hidden_states_mix = self.time_conv[i](hidden_states_mix)
                hidden_states_mix = rearrange(hidden_states_mix, 'B C T H W -> (B T) (H W) C')

            hidden_states_mix = temporal_block(
                hidden_states_mix,
                num_frames=num_frames,
                encoder_hidden_states=time_context,
            )
            hidden_states = self.time_mixer(
                x_spatial=hidden_states,
                x_temporal=hidden_states_mix,
                image_only_indicator=image_only_indicator,
            )

        # 3. Output
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual

        if not return_dict:
            return (output,)

        return TransformerTemporalModelOutput(sample=output)


class TrainableTemporalTimesteps(torch.nn.Module):
    def __init__(
        self,
        max_id: int,
        repeat_length: int,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        num_frames: int,
    ):
        super().__init__()
        timesteps = PositionalID(max_id, repeat_length)(num_frames)
        embeddings = get_timestep_embedding(timesteps, num_channels, flip_sin_to_cos, downscale_freq_shift)
        self.embeddings = torch.nn.Parameter(embeddings)

    def forward(self, timesteps):
        t_emb = self.embeddings[timesteps]
        return t_emb


class PositionalID(torch.nn.Module):
    def __init__(self, max_id=25, repeat_length=20):
        super().__init__()
        self.max_id = max_id
        self.repeat_length = repeat_length

    def frame_id_to_position_id(self, frame_id):
        if frame_id < self.max_id:
            position_id = frame_id
        else:
            position_id = (frame_id - self.max_id) % (self.repeat_length * 2)
            if position_id < self.repeat_length:
                position_id = self.max_id - 2 - position_id
            else:
                position_id = self.max_id - 2 * self.repeat_length + position_id
        return position_id

    def forward(self, num_frames, pivot_frame_id=0):
        position_ids = [self.frame_id_to_position_id(abs(i - pivot_frame_id)) for i in range(num_frames)]
        position_ids = torch.IntTensor(position_ids)
        return position_ids


def _init_time_conv(conv):
    for i in range(len(conv)):
        weight_shape = conv[i].weight.shape
        weight = torch.zeros(weight_shape)
        weight[:, :, 1, 1, 1] = torch.eye(weight_shape[0], weight_shape[1])
        conv[i].weight = nn.Parameter(weight, requires_grad=True)
        nn.init.constant_(conv[i].bias, 0)


setattr(transformer_temporal, 'TransformerSpatioTemporalModel', TransformerSpatioTemporalModel)
setattr(unet_3d_blocks, 'TransformerSpatioTemporalModel', TransformerSpatioTemporalModel)
