import numpy as np
import torch
import torch.nn as nn
from diffusers.models.embeddings import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed_from_grid
from einops import rearrange


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """grid_size: int of the grid height and width return: pos_embed:
    [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/
    or w/o cls_token)"""
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        bias=True,
        with_pos_embed=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.height = height // patch_size
        self.width = width // patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None
        self.with_pos_embed = with_pos_embed
        if self.with_pos_embed:
            pos_embed = get_2d_sincos_pos_embed(embed_dim, (self.height, self.width))
            self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

    def forward(self, latent):
        latent = self.proj(latent)
        height, width = latent.shape[-2], latent.shape[-1]
        latent = rearrange(latent, 'b c h w -> b (h w) c')
        if self.norm is not None:
            latent = self.norm(latent)
        if self.with_pos_embed:
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (height, width))
                pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed
            pos_embed = pos_embed.to(latent.dtype)
            latent = latent + pos_embed
        return latent


class FrameEmbed(nn.Module):
    def __init__(self, num_frames, embed_dim):
        super().__init__()
        self.num_frames = num_frames
        pos = torch.arange(0, num_frames)
        pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
        self.register_buffer('pos_embed', torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)

    def forward(self, latent):
        num_frames = latent.shape[1]
        if self.num_frames != num_frames:
            pos = torch.arange(0, num_frames)
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], pos)
            pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(latent.device)
        else:
            pos_embed = self.pos_embed
        pos_embed = pos_embed.to(latent.dtype)
        return latent + pos_embed
