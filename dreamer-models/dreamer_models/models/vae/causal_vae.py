import copy

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch import nn


class CausalVAEModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        z_channels=4,
        hidden_size=128,
        ch_mult=(1, 2, 4, 4),
        num_res_block=2,
        attn_resolutions=[],
        resolution=256,
        dropout=0.0,
        time_compress=2,
        time_downsample_method='avg_pool',
        time_upsample_method=None,
        scaling_factor=0.18215,
        force_upcast=False,
    ):
        super().__init__()
        if isinstance(time_compress, int):
            time_compress = [2 if i < time_compress else 1 for i in range(len(ch_mult))]
        assert isinstance(time_compress, (list, tuple)) and len(time_compress) == len(ch_mult)
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_size=hidden_size,
            z_channels=z_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_block,
            attn_resolutions=attn_resolutions,
            resolution=resolution,
            dropout=dropout,
            time_compress=time_compress,
            time_downsample_method=time_downsample_method,
        )
        self.decoder = Decoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            out_channels=out_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_block,
            attn_resolutions=attn_resolutions,
            resolution=resolution,
            dropout=dropout,
            time_compress=time_compress,
            time_upsample_method=time_upsample_method,
        )
        self.quant_conv = CausalConv3d(2 * z_channels, 2 * z_channels, 1)
        self.post_quant_conv = CausalConv3d(z_channels, z_channels, 1)
        self.scaling_factor = scaling_factor
        self.force_upcast = force_upcast

        self.time_multiple = np.prod(time_compress)
        self.spatial_multiple = 2 ** (len(ch_mult) - 1)

        self.tile_sample_size = (4 * self.time_multiple + 1, 512, 512)
        self.tile_overlap_factor = 1.0 / 8
        self.use_tiling = False
        self.use_slicing = False

    def load_state_dict(self, state_dict, *args, **kwargs):
        is_diffusers = False
        for key in state_dict.keys():
            if 'down_blocks' in key:
                is_diffusers = True
                break
        if is_diffusers:
            state_dict = _convert_diffusers_state_dict(self.state_dict(), state_dict)
        return super().load_state_dict(state_dict, *args, **kwargs)

    def forward(self, input, sample_posterior=True, generator=None):
        posterior = self.encode(input).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample
        return dec, posterior

    def encode(self, x, return_dict=True):
        x_ndim = x.ndim
        if x_ndim == 4:
            x = rearrange(x, '(b f) c h w -> b c f h w', b=1)

        if self.use_slicing and x.shape[0] > 1:
            moments_list = []
            for x_slice in x.split(1):
                if self.use_tiling:
                    moments = self.tiled_encode(x_slice)
                else:
                    h = self.encoder(x_slice)
                    moments = self.quant_conv(h)
                moments_list.append(moments)
            moments = torch.cat(moments_list)
        else:
            if self.use_tiling:
                moments = self.tiled_encode(x)
            else:
                h = self.encoder(x)
                moments = self.quant_conv(h)

        if x_ndim == 4:
            moments = rearrange(moments, 'b c f h w -> (b f) c h w', b=1)

        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z, return_dict=True):
        z_ndim = z.ndim
        if z_ndim == 4:
            z = rearrange(z, '(b f) c h w -> b c f h w', b=1)

        if self.use_slicing and z.shape[0] > 1:
            dec_list = []
            for z_slice in z.split(1):
                if self.use_tiling:
                    dec = self.tiled_decode(z_slice)
                else:
                    h = self.post_quant_conv(z_slice)
                    dec = self.decoder(h)
                dec_list.append(dec)
            dec = torch.cat(dec_list)
        else:
            if self.use_tiling:
                dec = self.tiled_decode(z)
            else:
                h = self.post_quant_conv(z)
                dec = self.decoder(h)

        if z_ndim == 4:
            dec = rearrange(dec, 'b c f h w -> (b f) c h w', b=1)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def tiled_encode(self, x):
        overlap_factor = self.tile_overlap_factor
        sample_size_t, sample_size_h, sample_size_w = self.tile_sample_size

        sample_size_t = sample_size_t if sample_size_t > 0 else x.shape[2]
        sample_size_h = sample_size_h if sample_size_h > 0 else x.shape[3]
        sample_size_w = sample_size_w if sample_size_w > 0 else x.shape[4]

        if sample_size_t >= x.shape[2] and sample_size_h >= x.shape[3] and sample_size_w >= x.shape[4]:
            h = self.encoder(x)
            moments = self.quant_conv(h)
            return moments

        latent_size_t = int((sample_size_t - 1) / self.time_multiple) + 1
        latent_size_h = int(sample_size_h / self.spatial_multiple)
        latent_size_w = int(sample_size_w / self.spatial_multiple)

        overlap_size_t = sample_size_t
        row_limit_t = latent_size_t

        overlap_size_h = max(int(sample_size_h * (1 - overlap_factor)), 1)
        blend_extent_h = max(int(latent_size_h * overlap_factor), 0)
        row_limit_h = latent_size_h - blend_extent_h

        overlap_size_w = max(int(sample_size_w * (1 - overlap_factor)), 1)
        blend_extent_w = max(int(latent_size_w * overlap_factor), 0)
        row_limit_w = latent_size_w - blend_extent_w

        # Split the image into tiles and encode them separately.
        rows_t = []
        for t in range(0, x.shape[2], overlap_size_t):
            rows = []
            for i in range(0, x.shape[3], overlap_size_h):
                row = []
                for j in range(0, x.shape[4], overlap_size_w):
                    tile = x[
                        :,
                        :,
                        t : t + sample_size_t,
                        i : i + sample_size_h,
                        j : j + sample_size_w,
                    ]
                    tile = self.encoder(tile)
                    tile = self.quant_conv(tile)
                    row.append(tile)
                    if j + sample_size_w >= x.shape[4]:
                        break
                rows.append(row)
                if i + sample_size_h >= x.shape[3]:
                    break
            rows_t.append(rows)

        result_rows_t = []
        for t, rows in enumerate(rows_t):
            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile
                    # to the current tile and add the current tile to the result row
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_extent_h)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_extent_w)
                    row_limit_i = row_limit_h if i != len(rows) - 1 else tile.shape[3]
                    row_limit_j = row_limit_w if j != len(row) - 1 else tile.shape[4]
                    result_row.append(tile[:, :, :row_limit_t, :row_limit_i, :row_limit_j])
                result_rows.append(torch.cat(result_row, dim=4))
            result_rows_t.append(torch.cat(result_rows, dim=3))

        moments = torch.cat(result_rows_t, dim=2)

        assert moments.shape[3] * self.spatial_multiple == x.shape[3]
        assert moments.shape[4] * self.spatial_multiple == x.shape[4]

        return moments

    def tiled_decode(self, z):
        overlap_factor = self.tile_overlap_factor
        sample_size_t, sample_size_h, sample_size_w = self.tile_sample_size

        sample_size_t = sample_size_t if sample_size_t > 0 else int((z.shape[2] - 1) * self.time_multiple + 1)
        sample_size_h = sample_size_h if sample_size_h > 0 else int(z.shape[3] * self.spatial_multiple)
        sample_size_w = sample_size_w if sample_size_w > 0 else int(z.shape[4] * self.spatial_multiple)

        latent_size_t = int((sample_size_t - 1) / self.time_multiple) + 1
        latent_size_h = int(sample_size_h / self.spatial_multiple)
        latent_size_w = int(sample_size_w / self.spatial_multiple)

        if latent_size_t >= z.shape[2] and latent_size_h >= z.shape[3] and latent_size_w >= z.shape[4]:
            z = self.post_quant_conv(z)
            dec = self.decoder(z)
            return dec

        overlap_size_t = latent_size_t
        row_limit_t = sample_size_t

        overlap_size_h = max(int(latent_size_h * (1 - overlap_factor)), 1)
        blend_extent_h = max(int(sample_size_h * overlap_factor), 0)
        row_limit_h = sample_size_h - blend_extent_h

        overlap_size_w = max(int(latent_size_w * (1 - overlap_factor)), 1)
        blend_extent_w = max(int(sample_size_w * overlap_factor), 0)
        row_limit_w = sample_size_w - blend_extent_w

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows_t = []
        for t in range(0, z.shape[2], overlap_size_t):
            rows = []
            for i in range(0, z.shape[3], overlap_size_h):
                row = []
                for j in range(0, z.shape[4], overlap_size_w):
                    tile = z[
                        :,
                        :,
                        t : t + latent_size_t,
                        i : i + latent_size_h,
                        j : j + latent_size_w,
                    ]
                    tile = self.post_quant_conv(tile)
                    decoded = self.decoder(tile)
                    row.append(decoded)
                    if j + latent_size_w >= z.shape[4]:
                        break
                rows.append(row)
                if i + latent_size_h >= z.shape[3]:
                    break
            rows_t.append(rows)

        result_rows_t = []
        for t, rows in enumerate(rows_t):
            result_rows = []
            for i, row in enumerate(rows):
                result_row = []
                for j, tile in enumerate(row):
                    # blend the above tile and the left tile
                    # to the current tile and add the current tile to the result row
                    if i > 0:
                        tile = self.blend_v(rows[i - 1][j], tile, blend_extent_h)
                    if j > 0:
                        tile = self.blend_h(row[j - 1], tile, blend_extent_w)
                    row_limit_i = row_limit_h if i != len(rows) - 1 else tile.shape[3]
                    row_limit_j = row_limit_w if j != len(row) - 1 else tile.shape[4]
                    result_row.append(tile[:, :, :row_limit_t, :row_limit_i, :row_limit_j])
                result_rows.append(torch.cat(result_row, dim=4))
            result_rows_t.append(torch.cat(result_rows, dim=3))

        dec = torch.cat(result_rows_t, dim=2)

        assert z.shape[3] * self.spatial_multiple == dec.shape[3]
        assert z.shape[4] * self.spatial_multiple == dec.shape[4]

        return dec

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def enable_tiling(self):
        self.use_tiling = True

    def disable_tiling(self):
        self.use_tiling = False

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def enable_gradient_checkpointing(self):
        for module in self.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        for module in self.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = False


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        z_channels,
        hidden_size,
        ch_mult,
        num_res_blocks,
        time_compress,
        attn_resolutions=[],
        resolution=256,
        dropout=0.0,
        double_z=True,
        time_downsample_method='avg_pool',
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = CausalConv3d(in_channels, hidden_size, kernel_size=3, stride=1, padding=1)
        # downsample
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = hidden_size * in_ch_mult[i_level]
            block_out = hidden_size * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = SpatialDownsample2x(block_in, block_in)
                curr_res = curr_res // 2
            if time_compress[i_level] == 2:
                down.time_downsample = TimeDownsample2x(block_in, method=time_downsample_method)
            else:
                assert time_compress[i_level] == 1
            self.down.append(down)
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = CausalConv3d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.gradient_checkpointing = False

    def _forward_down_block(self, h, i_level):
        for i_block in range(self.num_res_blocks):
            h = self.down[i_level].block[i_block](h)
            if len(self.down[i_level].attn) > 0:
                h = self.down[i_level].attn[i_block](h)
        if hasattr(self.down[i_level], 'downsample'):
            h = self.down[i_level].downsample(h)
        if hasattr(self.down[i_level], 'time_downsample'):
            h = self.down[i_level].time_downsample(h)
        return h

    def _forward_mid_block(self, h):
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        return h

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            if self.training and self.gradient_checkpointing:
                h = torch.utils.checkpoint.checkpoint(self._forward_down_block, h, i_level, use_reentrant=False)
            else:
                h = self._forward_down_block(h, i_level)
        if self.training and self.gradient_checkpointing:
            h = torch.utils.checkpoint.checkpoint(self._forward_mid_block, h, use_reentrant=False)
        else:
            h = self._forward_mid_block(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels,
        out_channels,
        hidden_size,
        ch_mult,
        num_res_blocks,
        time_compress,
        attn_resolutions=[],
        resolution=256,
        dropout=0.0,
        time_upsample_method=None,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        block_in = hidden_size * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, padding=1)
        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
        )
        # upsample
        reversed_time_compress = list(reversed(time_compress))
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = hidden_size * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = SpatialUpsample2x(block_in, block_in)
                curr_res = curr_res * 2
            if reversed_time_compress[i_level] == 2:
                up.time_upsample = TimeUpsample2x(block_in, method=time_upsample_method)
            else:
                assert reversed_time_compress[i_level] == 1
            self.up.insert(0, up)  # prepend to get consistent order
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3, padding=1)
        self.gradient_checkpointing = False

    def _forward_up_block(self, h, i_level):
        for i_block in range(self.num_res_blocks + 1):
            h = self.up[i_level].block[i_block](h)
            if len(self.up[i_level].attn) > 0:
                h = self.up[i_level].attn[i_block](h)
        if hasattr(self.up[i_level], 'upsample'):
            h = self.up[i_level].upsample(h)
        if hasattr(self.up[i_level], 'time_upsample'):
            h = self.up[i_level].time_upsample(h)
        return h

    def _forward_mid_block(self, h):
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        return h

    def forward(self, z):
        h = self.conv_in(z)
        if self.training and self.gradient_checkpointing:
            h = torch.utils.checkpoint.checkpoint(self._forward_mid_block, h, use_reentrant=False)
        else:
            h = self._forward_mid_block(h)
        for i_level in reversed(range(self.num_resolutions)):
            if self.training and self.gradient_checkpointing:
                h = torch.utils.checkpoint.checkpoint(self._forward_up_block, h, i_level, use_reentrant=False)
            else:
                h = self._forward_up_block(h, i_level)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, time_causal_padding=None):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        stride = cast_tuple(stride, 3)
        padding = list(cast_tuple(padding, 3))
        padding[0] = 0
        if time_causal_padding is None:
            time_causal_padding = kernel_size[0] - 1
        self.kernel_size = kernel_size
        self.time_causal_padding = time_causal_padding
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.time_causal_padding, 1, 1))
        y = torch.concatenate((first_frame_pad, x), dim=2)
        y = self.conv(y)
        return y


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=3, padding=1)
            else:
                self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, t, h, w = q.shape
        q = rearrange(q, 'b c t h w -> (b t) (h w) c')
        k = rearrange(k, 'b c t h w -> (b t) c (h w)')
        v = rearrange(v, 'b c t h w -> (b t) c (h w)')

        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        w_ = w_.permute(0, 2, 1)

        h_ = torch.bmm(v, w_)
        h_ = rearrange(h_, '(b t) c (h w) -> b c t h w', b=b, t=t, h=h, w=w)
        h_ = self.proj_out(h_)

        return x + h_


class TemporalAttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.k = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.v = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)
        self.proj_out = nn.Conv3d(in_channels, in_channels, kernel_size=1, stride=1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, t, h, w = q.shape
        q = rearrange(q, 'b c t h w -> (b h w) t c')
        k = rearrange(k, 'b c t h w -> (b h w) c t')
        v = rearrange(v, 'b c t h w -> (b h w) c t')

        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)
        w_ = w_.permute(0, 2, 1)

        h_ = torch.bmm(v, w_)
        h_ = rearrange(h_, '(b h w) c t -> b c t h w', b=b, h=h, w=w)
        h_ = self.proj_out(h_)

        return x + h_


class SpatialDownsample2x(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=0)

    def forward(self, x):
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        pad = (0, 1, 0, 1)
        x = torch.nn.functional.pad(x, pad, mode='constant', value=0)
        x = self.conv(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x


class SpatialUpsample2x(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=(1, 2, 2), mode='nearest')
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = F.interpolate(x, scale_factor=(2, 2), mode='nearest')
        x = self.conv(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)
        return x


class TimeDownsample2x(nn.Module):
    def __init__(self, in_channels, kernel_size=3, method='avg_pool'):
        super().__init__()
        self.in_channels = in_channels
        self.method = method
        if method == 'avg_pool':
            self.kernel_size = (kernel_size, 1, 1)
            self.avg_pool = nn.AvgPool3d(self.kernel_size, stride=(2, 1, 1))

        elif method == 'conv':
            self.kernel_size = cast_tuple(kernel_size, 3)
            self.conv = CausalConv3d(in_channels, in_channels, self.kernel_size, stride=(2, 1, 1), padding=1)
            self.reset_parameters()

        elif method == 'attn':
            self.kernel_size = cast_tuple(kernel_size, 3)
            self.conv = nn.Sequential(
                ResnetBlock3D(in_channels, in_channels),
                TemporalAttnBlock3D(in_channels),
                CausalConv3d(in_channels, in_channels, self.kernel_size, stride=(2, 1, 1), padding=1),
            )

        else:
            assert False

    def reset_parameters(self):
        device = self.conv.conv.weight.device
        dtype = self.conv.conv.weight.dtype
        weight = torch.zeros((self.in_channels, self.in_channels, *self.kernel_size), device=device, dtype=dtype)
        data = torch.eye(self.in_channels).unsqueeze(-1).repeat(1, 1, self.kernel_size[0]) / self.kernel_size[0]
        weight[:, :, :, self.kernel_size[1] // 2, self.kernel_size[2] // 2] = data
        self.conv.conv.weight.data[:, :] = weight
        if self.conv.conv.bias is not None:
            nn.init.constant_(self.conv.conv.bias, 0)

    def forward(self, x):
        if self.method == 'avg_pool':
            first_frame_pad = x[:, :, :1, :, :].repeat((1, 1, self.kernel_size[0] - 1, 1, 1))
            x = torch.concatenate((first_frame_pad, x), dim=2)
            x = self.avg_pool(x)
        else:
            x = self.conv(x)
        return x


class TimeUpsample2x(nn.Module):
    def __init__(self, in_channels, kernel_size=3, method=None):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.method = method
        if self.method == 'conv':
            self.conv = CausalConv3d(in_channels, in_channels, self.kernel_size, stride=1, padding=1)
            self.reset_parameters()

        elif self.method == 'attn':
            self.conv = nn.Sequential(
                ResnetBlock3D(in_channels, in_channels),
                TemporalAttnBlock3D(in_channels),
                CausalConv3d(in_channels, in_channels, self.kernel_size, stride=1, padding=1),
            )

        else:
            assert self.method is None

    def reset_parameters(self):
        device = self.conv.conv.weight.device
        dtype = self.conv.conv.weight.dtype
        weight = torch.zeros((self.in_channels, self.in_channels, *self.kernel_size), device=device, dtype=dtype)
        data = torch.eye(self.in_channels).unsqueeze(-1).repeat(1, 1, self.kernel_size[0]) / self.kernel_size[0]
        weight[:, :, :, self.kernel_size[1] // 2, self.kernel_size[2] // 2] = data
        self.conv.conv.weight.data[:, :] = weight
        if self.conv.conv.bias is not None:
            nn.init.constant_(self.conv.conv.bias, 0)

    def forward(self, x):
        if x.size(2) > 1:
            x, x_ = x[:, :, :1], x[:, :, 1:]
            x_ = F.interpolate(x_, scale_factor=(2, 1, 1), mode='trilinear')
            x = torch.concat([x, x_], dim=2)
        if self.method is not None:
            x = self.conv(x)
        return x


def nonlinearity(x):
    return x * torch.sigmoid(x)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def _convert_diffusers_state_dict(state_dict, d_state_dict):
    num_up_blocks = 0
    for key in state_dict:
        if key.startswith('decoder.up'):
            up_idx = int(key[len('decoder.up.')])
            num_up_blocks = max(up_idx + 1, num_up_blocks)
    keys = {
        'encoder': {
            'down_blocks': [],
            'mid_block': {
                'attentions': [],
                'resnets': [],
            },
            'others': [],
        },
        'decoder': {
            'up_blocks': {k: [] for k in range(num_up_blocks)},
            'mid_block': {
                'attentions': [],
                'resnets': [],
            },
            'others': [],
        },
        'others': [],
    }
    d_keys = copy.deepcopy(keys)
    for key in state_dict.keys():
        if key.startswith('encoder'):
            if key.startswith('encoder.down'):
                if 'time_downsample' not in key:
                    keys['encoder']['down_blocks'].append(key)
            elif key.startswith('encoder.mid'):
                if key.startswith('encoder.mid.attn'):
                    keys['encoder']['mid_block']['attentions'].append(key)
                elif key.startswith('encoder.mid.block'):
                    keys['encoder']['mid_block']['resnets'].append(key)
                else:
                    assert False
            else:
                keys['encoder']['others'].append(key)
        elif key.startswith('decoder'):
            if key.startswith('decoder.up'):
                if 'time_upsample' not in key:
                    up_idx = int(key[len('decoder.up.')])
                    keys['decoder']['up_blocks'][up_idx].append(key)
            elif key.startswith('decoder.mid'):
                if key.startswith('decoder.mid.attn'):
                    keys['decoder']['mid_block']['attentions'].append(key)
                elif key.startswith('decoder.mid.block'):
                    keys['decoder']['mid_block']['resnets'].append(key)
                else:
                    assert False
            else:
                keys['decoder']['others'].append(key)
        else:
            keys['others'].append(key)
    for key in d_state_dict.keys():
        if key.startswith('encoder'):
            if key.startswith('encoder.down_blocks'):
                d_keys['encoder']['down_blocks'].append(key)
            elif key.startswith('encoder.mid_block'):
                if key.startswith('encoder.mid_block.attentions'):
                    d_keys['encoder']['mid_block']['attentions'].append(key)
                elif key.startswith('encoder.mid_block.resnets'):
                    d_keys['encoder']['mid_block']['resnets'].append(key)
                else:
                    assert False
            else:
                d_keys['encoder']['others'].append(key)
        elif key.startswith('decoder'):
            if key.startswith('decoder.up_blocks'):
                up_idx = int(key[len('decoder.up_blocks.')])
                up_idx = num_up_blocks - 1 - up_idx
                d_keys['decoder']['up_blocks'][up_idx].append(key)
            elif key.startswith('decoder.mid_block'):
                if key.startswith('decoder.mid_block.attentions'):
                    d_keys['decoder']['mid_block']['attentions'].append(key)
                elif key.startswith('decoder.mid_block.resnets'):
                    d_keys['decoder']['mid_block']['resnets'].append(key)
                else:
                    assert False
            else:
                d_keys['decoder']['others'].append(key)
        else:
            d_keys['others'].append(key)

    def _check(keys_a, keys_b):
        if isinstance(keys_a, list):
            assert len(keys_a) == len(keys_b)
        elif isinstance(keys_a, dict):
            for _ in keys_a:
                _check(keys_a[_], keys_b[_])
        else:
            assert False

    _check(keys, d_keys)

    def _concat(keys_a):
        new_keys = []
        for key in keys_a:
            if isinstance(keys_a[key], list):
                new_keys.extend(keys_a[key])
            elif isinstance(keys_a[key], dict):
                new_keys.extend(_concat(keys_a[key]))
            else:
                assert False
        return new_keys

    keys = _concat(keys)
    d_keys = _concat(d_keys)

    assert len(keys) == len(d_keys)
    for key, d_key in zip(keys, d_keys):
        val = state_dict[key]
        d_val = d_state_dict[d_key]
        if val.shape != d_val.shape:
            if val.ndim == d_val.ndim + 1:
                d_val = d_val.unsqueeze(2).repeat(1, 1, val.shape[2], 1, 1) / val.shape[2]
            elif np.prod(val.shape) == np.prod(d_val.shape):
                d_val = d_val.reshape(val.shape)
            else:
                assert False
            assert val.shape == d_val.shape
        state_dict[key][:] = d_val

    return state_dict
