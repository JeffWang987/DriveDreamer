import math

import torch
import torch.nn as nn


class EDMLoss(nn.Module):
    """Loss in the paper "Elucidating the Design Space of Diffusion-Based
    Generative Models" (EDM)."""

    def __init__(self, sigma_method=1, p_mean=-1.2, p_std=1.2, sigma_data=0.5):
        super().__init__()
        self.sigma_method = sigma_method
        self.p_mean = p_mean
        self.p_std = p_std
        self.sigma_data = sigma_data

    def add_noise(self, latents, num_frames=1):
        assert latents.shape[0] % num_frames == 0
        batch_size = latents.shape[0] // num_frames
        if self.sigma_method == 1:
            rnd_normal = torch.randn((batch_size,), device=latents.device)
            sigma = (rnd_normal * self.p_std + self.p_mean).exp()
        elif self.sigma_method == 2:
            sigma = rand_cosine_interpolated(shape=(batch_size,)).to(latents.device)
        else:
            assert False
        sigma_r = sigma[:, None, None, None].repeat_interleave(num_frames, dim=0)
        c_in = 1 / (sigma_r**2 + self.sigma_data**2).sqrt()
        noise = torch.randn_like(latents)
        noisy_latents = latents + noise * sigma_r
        inp_noisy_latents = noisy_latents * c_in
        timesteps = 0.25 * sigma.log()
        timesteps = timesteps.repeat_interleave(num_frames, dim=0)
        self.sigma = sigma.reshape((batch_size, 1))
        self.latents = latents.reshape((batch_size, -1))
        self.noisy_latents = noisy_latents.reshape((batch_size, -1))
        return inp_noisy_latents, timesteps

    def forward(self, pred_latents):
        c_skip = self.sigma_data**2 / (self.sigma**2 + self.sigma_data**2)
        c_out = self.sigma * self.sigma_data / (self.sigma**2 + self.sigma_data**2).sqrt()
        pred_latents = pred_latents.reshape((self.sigma.shape[0], -1))
        denoised_latents = c_skip * self.noisy_latents + c_out * pred_latents
        weight = (self.sigma**2 + self.sigma_data**2) / (self.sigma * self.sigma_data) ** 2
        loss = weight * (denoised_latents - self.latents) ** 2
        loss = torch.mean(loss, dim=1)
        return loss


# copy from https://github.com/crowsonkb/k-diffusion.git
def stratified_uniform(shape, group=0, groups=1, dtype=None, device=None):
    """Draws stratified samples from a uniform distribution."""
    if groups <= 0:
        raise ValueError(f'groups must be positive, got {groups}')
    if group < 0 or group >= groups:
        raise ValueError(f'group must be in [0, {groups})')
    n = shape[-1] * groups
    offsets = torch.arange(group, n, groups, dtype=dtype, device=device)
    u = torch.rand(shape, dtype=dtype, device=device)
    return (offsets + u) / n


def rand_cosine_interpolated(
    shape,
    image_d=64,
    noise_d_low=32,
    noise_d_high=64,
    sigma_data=0.5,
    min_value=0.002,
    max_value=700,
    device='cpu',
    dtype=torch.float32,
):
    """Draws samples from an interpolated cosine timestep distribution (from
    simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min)))

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = stratified_uniform(shape, group=0, groups=1, dtype=dtype, device=device)
    logsnr = logsnr_schedule_cosine_interpolated(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data
