import json
import os

import torch
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from .utils import add_control_model_name, repeat_data


class ClipMixin:
    def load_clip_model(self, pretrained_model_path, **kwargs):
        kwargs.setdefault('torch_dtype', self.dtype)
        self.clip_image_processor = CLIPImageProcessor()
        config_path = os.path.join(pretrained_model_path, 'config.json')
        config = json.load(open(config_path, 'r'))
        class_name = config['_class_name']
        if class_name == 'CLIPVisionModelWithProjection':
            self.clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, **kwargs)
        else:
            assert False
        add_control_model_name('clip_image_encoder')

    def get_timesteps(self, num_inference_steps, strength, denoising_start=None):
        # get the original timestep using init_timestep
        if denoising_start is None:
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
        else:
            t_start = 0
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        # Strength is irrelevant if we directly request a timestep to start at;
        # that is, strength is determined by the denoising_start instead.
        if denoising_start is not None:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_start * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
            if self.scheduler.order == 2 and num_inference_steps % 2 == 0:
                # if the scheduler is a 2nd order scheduler we might have to do +1
                # because `num_inference_steps` might be even given that every timestep
                # (except the highest one) is duplicated. If `num_inference_steps` is even it would
                # mean that we cut the timesteps in the middle of the denoising step
                # (between 1st and 2nd devirative) which leads to incorrect results. By adding 1
                # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
                num_inference_steps = num_inference_steps + 1
            # because t_n+1 >= t_n, we slice the timesteps starting from the end
            timesteps = timesteps[-num_inference_steps:]
            return timesteps, num_inference_steps
        return timesteps, num_inference_steps - t_start

    def encode_clip_image(
        self,
        image,
        batch_size=1,
        num_images_per_prompt=1,
        num_frames=None,
        do_classifier_free_guidance=False,
    ):
        if image is None:
            return None
        device = self._execution_device
        pixel_values = self.clip_image_processor(
            images=image,
            size=dict(height=224, width=224),
            do_center_crop=False,
            return_tensors='pt',
        ).pixel_values
        pixel_values = pixel_values.to(device=device, dtype=self.clip_image_encoder.dtype)
        image_embeddings = self.clip_image_encoder(pixel_values).image_embeds.unsqueeze(1)
        image_embeddings = repeat_data(
            image_embeddings,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            num_frames=num_frames,
        )
        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])
        return image_embeddings
