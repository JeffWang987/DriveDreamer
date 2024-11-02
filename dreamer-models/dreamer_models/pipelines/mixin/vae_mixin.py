import json
import os

import torch
from diffusers import AutoencoderKL, AutoencoderKLTemporalDecoder
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img import randn_tensor

from ...models import CausalVAEModel
from .utils import repeat_data


class VaeMixin:
    def load_vae_model(self, pretrained_model_path, **kwargs):
        kwargs.setdefault('torch_dtype', self.dtype)
        config_path = os.path.join(pretrained_model_path, 'config.json')
        config = json.load(open(config_path, 'r'))
        class_name = config['_class_name']
        if class_name == 'AutoencoderKL':
            vae = AutoencoderKL.from_pretrained(pretrained_model_path, **kwargs)
            self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        elif class_name == 'AutoencoderKLTemporalDecoder':
            vae = AutoencoderKLTemporalDecoder.from_pretrained(pretrained_model_path, **kwargs)
            self.vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        elif class_name == 'CausalVAEModel':
            vae = CausalVAEModel.from_pretrained(pretrained_model_path, **kwargs)
            self.vae_scale_factor = vae.spatial_multiple
        else:
            assert False
        self.vae = vae

    def encode_vae_image(
        self,
        image,
        height,
        width,
        batch_size=1,
        num_images_per_prompt=1,
        num_frames=None,
        do_classifier_free_guidance=False,
        sample_mode='sample',
        add_noise=True,
        timestep=None,
        generator=None,
    ):
        if image is None:
            return None
        device = self._execution_device
        image = self.image_processor.preprocess(image, height, width)
        image = image.to(device=device)
        needs_upcasting = self.vae.config.force_upcast
        if needs_upcasting:
            vae_dtype = self.vae.dtype
            image = image.float()
            self.vae.to(dtype=torch.float32)
        else:
            image = image.to(self.vae.dtype)
        latent_dist = self.vae.encode(image).latent_dist
        if sample_mode == 'sample':
            latents = latent_dist.sample(generator)
        elif sample_mode == 'argmax':
            latents = latent_dist.mode()
        else:
            assert False
        if needs_upcasting:
            self.vae.to(vae_dtype)
        latents = latents * self.vae.config.scaling_factor
        latents = repeat_data(
            latents,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            num_frames=num_frames,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        if add_noise:
            noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
            latents = self.scheduler.add_noise(latents, noise, timestep)
        return latents
