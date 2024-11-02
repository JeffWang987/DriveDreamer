# Copyright 2023 The GLIGEN Authors and HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Any, Callable, Dict, List, Optional, Union

import PIL
import torch
from accelerate import cpu_offload_with_hook
from compel import Compel
from diffusers import StableDiffusionPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg, retrieve_timesteps

from ...models import UNet2DConditionModel, UNet3DConditionModel, UNetSpatioTemporalConditionModel
from ..mixin import (
    ClipMixin,
    DriveDreamerMixin,
    LoraMixin,
    PromptTravelMixin,
    UNetMixin,
    VaeMixin,
    get_control_model_names,
)


class StableDiffusionControlPipeline(
    StableDiffusionPipeline,
    DriveDreamerMixin,
    ClipMixin,
    LoraMixin,
    UNetMixin,
    VaeMixin,
    PromptTravelMixin,
):
    r"""Pipeline for text-to-image generation using Stable Diffusion with
    Control."""

    def to(self, device=None, dtype=None):
        if device is None and dtype is None:
            return self
        control_model_names = get_control_model_names()
        for model_name in control_model_names:
            model = getattr(self, model_name, None)
            if model is not None:
                model.to(device, dtype)
        super().to(device, dtype)

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = 'cuda'):
        super().enable_model_cpu_offload(gpu_id, device)
        device = self._all_hooks[0].hook.execution_device
        control_model_names = get_control_model_names()
        for model_name in control_model_names:
            model = getattr(self, model_name, None)
            if model is not None:
                _, hook = cpu_offload_with_hook(model, device)
                self._all_hooks.append(hook)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        unet_model_path=None,
        unet3d_model_path=None,
        clip_model_path=None,
        drivedreamer_model_path=None,
        lora_model_path=None,
        **kwargs
    ):
        pipe_kwargs = dict()
        if unet_model_path is None:
            unet_model_path = os.path.join(model_path, 'unet')
        unet_type = kwargs.pop('unet_type', 'UNet2DConditionModel')
        unet_from_2d_to_3d = kwargs.pop('unet_from_2d_to_3d', False)
        num_frames = kwargs.pop('num_frames', 32)
        torch_dtype = kwargs.pop('torch_dtype', torch.float32)
        if unet_type == 'UNet2DConditionModel':
            unet = UNet2DConditionModel.from_pretrained(unet_model_path, **kwargs)
            if unet_from_2d_to_3d:
                unet = UNet3DConditionModel.from_unet(
                    unet, dtype_for_2d=torch_dtype, 
                    dtype_for_3d=torch_dtype, 
                    num_frames=num_frames,
                    unet3d_model_path=unet3d_model_path,
                )
        elif unet_type == 'UNet3DConditionModel':
            unet = UNet3DConditionModel.from_pretrained(unet3d_model_path, **kwargs)
        elif unet_type == 'UNetSpatioTemporalConditionModel':
            unet = UNetSpatioTemporalConditionModel.from_pretrained(unet3d_model_path, **kwargs)
            
        pipe_kwargs['unet'] = unet
        pipe_kwargs['torch_dtype'] = torch_dtype
        pipe_kwargs.update(kwargs)
        kwargs.pop('safety_checker')
        pipe = super(cls, StableDiffusionControlPipeline).from_pretrained(model_path, **pipe_kwargs)
        if clip_model_path is not None:
            pipe.load_clip_model(clip_model_path, **kwargs)
        if drivedreamer_model_path is not None:
            pipe.load_drivedreamer_model(drivedreamer_model_path, **kwargs)
        if lora_model_path is not None:
            pipe.load_lora_model(lora_model_path, **kwargs)
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = 'pil',
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ['latents'],
        # image
        image: Optional[PipelineImageInput] = None,
        strength: float = 0.8,
        # clip image
        clip_image: Optional[PipelineImageInput] = None,
        # drivedreamer
        drivedreamer_image: Optional[PipelineImageInput] = None,
        drivedreamer_boxes: Optional[List[List[float]]] = None,
        drivedreamer_box_texts: Optional[List[str]] = None,
        drivedreamer_box_images: Optional[List[PIL.Image.Image]] = None,
        drivedreamer_max_objs: Optional[int] = 30,
        # video
        num_frames: Optional[int] = None,
        video_chunk_size: Optional[int] = None,
        # others
        seed: Optional[int] = -1,
    ):
        r"""The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        callback = None
        callback_steps = None

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        device = self._execution_device

        if seed > 0:
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)

        if prompt is not None or prompt_embeds is not None:
            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            lora_scale = (
                self.cross_attention_kwargs.get('scale', None) if self.cross_attention_kwargs is not None else None
            )

            if not hasattr(self, 'prompt_compel'):
                self.prompt_compel = Compel(
                    tokenizer=self.tokenizer,
                    text_encoder=self.text_encoder,
                    device=device,
                )
            assert prompt is not None
            prompt_embeds = self.prompt_compel(prompt)
            if negative_prompt is not None:
                negative_prompt_embeds = self.prompt_compel(negative_prompt)

            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
            )
            prompt_embeds = self.process_prompt_embeds(prompt_embeds, prompt, num_frames)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            if self.do_classifier_free_guidance:
                prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if num_frames is not None and len(negative_prompt_embeds) == 1:
                prompt_embeds = prompt_embeds.repeat_interleave(num_frames, dim=0)

        elif clip_image is not None:
            if isinstance(clip_image, list):
                batch_size = len(clip_image)
            else:
                batch_size = 1
            clip_image_embeds = self.encode_clip_image(
                image=clip_image,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
            prompt_embeds = clip_image_embeds

        else:
            assert False
        prompt_embeds = prompt_embeds.to(self.dtype)

        if num_frames is not None:
            if video_chunk_size is None:
                video_chunk_size = num_frames
            video_chunk_size = min(video_chunk_size, num_frames)
            if batch_size == num_frames:
                batch_size = 1
            assert batch_size == 1 and num_images_per_prompt == 1

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        if image is not None:
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
            latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
            latents = self.encode_vae_image(
                image=image,
                timestep=latent_timestep,
                height=height,
                width=width,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
                generator=generator,
            )
        elif num_frames is not None:
            latents_chunk_size = video_chunk_size
            latents = self.prepare_latents(
                batch_size=latents_chunk_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=self.dtype,
                device=device,
                generator=generator,
                latents=latents,
            )
            repeat_by = num_frames // latents_chunk_size
            if repeat_by > 1:
                latents = torch.cat([latents] * repeat_by)
            if latents.shape[0] != num_frames:
                latents = torch.cat([latents, latents[: (num_frames - latents.shape[0])]])
        else:
            latents = self.prepare_latents(
                batch_size=batch_size * num_images_per_prompt,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                dtype=self.dtype,
                device=device,
                generator=generator,
                latents=latents,
            )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        added_cond_kwargs = None

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # DriveDreamer
        drivedreamer_outputs = self.forward_DriveDreamer(
            image=drivedreamer_image,
            boxes=drivedreamer_boxes,
            box_texts=drivedreamer_box_texts,
            box_images=drivedreamer_box_images,
            height=height,
            width=width,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            max_objs=drivedreamer_max_objs,
            generator=generator,
        )
        objs = drivedreamer_outputs.get('objs', None)
        if objs is not None:
            if cross_attention_kwargs is None:
                cross_attention_kwargs = dict()
            cross_attention_kwargs.update(gligen=dict(objs=objs))

        self.set_num_frames(video_chunk_size)

        while True:
            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            self._num_timesteps = len(timesteps)
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                    if drivedreamer_image is not None:
                        latent_model_input = torch.cat((latent_model_input, drivedreamer_outputs['latents']), dim=1)

                    encoder_hidden_states = prompt_embeds

                    noise_pred = self.forward_unet(
                        latent_model_input=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                        cur_step=i,
                        num_frames=num_frames,
                        chunk_size=video_chunk_size,
                    )

                    # perform guidance
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(
                            noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                        latents = callback_outputs.pop('latents', latents)
                        prompt_embeds = callback_outputs.pop('prompt_embeds', prompt_embeds)
                        negative_prompt_embeds = callback_outputs.pop('negative_prompt_embeds', negative_prompt_embeds)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            step_idx = i // getattr(self.scheduler, 'order', 1)
                            callback(step_idx, t, latents)
            break

        if not output_type == 'latent':
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, self.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
