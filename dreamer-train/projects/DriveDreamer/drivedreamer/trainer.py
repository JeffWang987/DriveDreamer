import functools
import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import WEIGHTS_NAME
from dreamer_datasets import DefaultCollator, DefaultSampler, load_dataset
from dreamer_models import DriveDreamerModel, LoRAModel, UNet2DConditionModel, UNet3DConditionModel, UNetSpatioTemporalConditionModel

from dreamer_train import Trainer
from dreamer_train import utils as dt_utils
from dreamer_models import utils as dm_utils
from .transforms import DriveDreamerTransform
from .samplers import NuscVideoSampler


class DriveDreamerTrainer(Trainer):
    def get_dataloaders(self, data_config):
        dataset = load_dataset(data_config.data_or_config)
        batch_size = data_config.batch_size_per_gpu * self.num_processes * self.gradient_accumulation_steps
        filter_cfg = data_config.get('filter', None)
        if filter_cfg is not None:
            dataset.filter(**filter_cfg)
        transform_cfg = data_config.transform
        transform_type = transform_cfg.pop('type')
        if transform_type == 'DriveDreamerTransform':
            transform = DriveDreamerTransform(**transform_cfg)
        else:
            assert False
        sampler_cfg = data_config.get('sampler', {'type': 'DefaultSampler'})
        sampler_type = sampler_cfg.pop('type')
        if sampler_type == 'DefaultSampler':
            sampler = DefaultSampler(dataset, batch_size=batch_size, **sampler_cfg)
        elif sampler_type == 'NuscVideoSampler':
            sampler = NuscVideoSampler(dataset, batch_size=batch_size, logger=self.logger, **sampler_cfg)
        else:
            assert False
        dataset.set_transform(transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            collate_fn=DefaultCollator(),
            batch_size=data_config.batch_size_per_gpu,
            num_workers=data_config.num_workers,
        )
        return dataloader

    def get_models(self, model_config):
        pretrained = model_config.pretrained
        
        # VAE config
        vae_pretrained = model_config.get('vae_pretrained', pretrained)
        vae_subfolder = model_config.get('vae_subfolder', 'vae')
        vae_dtype = model_config.get('vae_dtype', 'float32')
        vae_dtype = getattr(torch, vae_dtype)
        self.vae = AutoencoderKL.from_pretrained(
            dm_utils.get_model_path(vae_pretrained),
            subfolder=vae_subfolder,
            revision=None,
            local_files_only=True,
        )
        self.vae.requires_grad_(False)
        self.vae.to(self.device, dtype=vae_dtype)
        
        # noise_scheduler config
        noise_scheduler_type = model_config.drivedreamer.pop('noise_scheduler_type', 'DDPMScheduler')
        if noise_scheduler_type == 'DDPMScheduler':
            self.noise_scheduler = DDPMScheduler.from_pretrained(
                dm_utils.get_model_path(pretrained),
                subfolder='scheduler',
                local_files_only=True,
            )
            
        # Unet config
        self.num_frames = model_config.drivedreamer.pop('num_frames', 1)
        self.num_cams = model_config.drivedreamer.pop('num_cams', 1)
        unet_type = model_config.drivedreamer.pop('unet_type', 'UNet2DConditionModel')
        self.unet_from_2d_to_3d = model_config.drivedreamer.pop('unet_from_2d_to_3d', False)
        if unet_type == 'UNet2DConditionModel':
            unet = UNet2DConditionModel.from_pretrained(
                dm_utils.get_model_path(pretrained),
                subfolder='unet',
                revision=None,
                local_files_only=True,
            )
            if self.unet_from_2d_to_3d:
                unet = UNet3DConditionModel.from_unet(
                    unet, dtype_for_2d=self.dtype, 
                    dtype_for_3d=torch.float32, 
                    tune_all_unet_params=model_config.drivedreamer.pop('tune_all_unet_params', False),
                    num_frames=self.num_frames,
                )
        elif unet_type == 'UNet3DConditionModel':
            unet = UNet3DConditionModel.from_pretrained(
                dm_utils.get_model_path(pretrained),
                subfolder='unet',
                revision=None,
                local_files_only=True,
            )
        elif unet_type == 'UNetSpatioTemporalConditionModel':
            unet = UNetSpatioTemporalConditionModel.from_pretrained(
                dm_utils.get_model_path(pretrained),
                subfolder='unet',
                revision=None,
                local_files_only=True,
            )
            
        unet.to(self.device)
        unet.train()
        if not self.unet_from_2d_to_3d:  # already done in UNet3DConditionModel.from_unet, thus skip here
            unet.requires_grad_(False)
            unet.to(dtype=self.dtype)
        
        # DriveDreamer config
        drivedreamer = DriveDreamerModel.from_unet(unet, **model_config.drivedreamer)
        drivedreamer.to_unet(unet)
        drivedreamer.to(self.device)
        drivedreamer.train()
        if self.activation_checkpointing:
            unet.enable_gradient_checkpointing()
            drivedreamer.enable_gradient_checkpointing()
        checkpoint = model_config.get('checkpoint', None)
        self.load_checkpoint(checkpoint, [unet, drivedreamer])
        self.unet = unet
        
        # Model-train config
        self.train_mode = model_config.get('train_mode', 'drivedreamer')  # lora or other
        self.model_name = self.train_mode
        if self.train_mode == 'drivedreamer':
            if self.unet_from_2d_to_3d:
                return nn.ModuleDict({
                    'drivedreamer': drivedreamer,
                    'unet': unet,
                })
            else:
                return nn.ModuleDict({
                    'drivedreamer': drivedreamer,
                })
        elif self.train_mode == 'lora':
            assert checkpoint is not None
            drivedreamer.requires_grad_(False)
            drivedreamer.to(dtype=self.dtype)
            self.drivedreamer = drivedreamer
            lora = LoRAModel.from_models([drivedreamer], rank=model_config.lora_rank)
            lora.to_models([drivedreamer], is_train=True)
            lora.to(self.device)
            lora.train()
            lora_checkpoint = model_config.get('lora_checkpoint', None)
            if lora_checkpoint is not None:
                lora_checkpoint = self.get_checkpoint(lora_checkpoint)
                lora_checkpoint = os.path.join(lora_checkpoint, 'lora', WEIGHTS_NAME)
                self.logger.info('Load lora from {}'.format(lora_checkpoint))
                state_dict = torch.load(lora_checkpoint, map_location='cpu')
                lora.load_state_dict(state_dict)
            return lora
        else:
            assert False

    def get_schedulers(self, scheduler):
        scheduler = get_scheduler(
            name=scheduler.name,
            optimizer=self.optimizer,
            num_warmup_steps=scheduler.num_warmup_steps * self.num_processes,
            num_training_steps=self.max_steps * self.num_processes,
        )
        return scheduler

    def forward_step(self, batch_dict):
        if self.train_mode == 'drivedreamer':
            if self.unet_from_2d_to_3d:
                unet = self.model.unet if isinstance(self.model, nn.ModuleDict) else self.model.module.unet
            else:
                unet = self.unet
            drivedreamer = self.model.drivedreamer if isinstance(self.model, nn.ModuleDict) else self.model.module.drivedreamer
            grounding_downsampler = functools.partial(drivedreamer, 'grounding_downsampler')
            position_net = functools.partial(drivedreamer, 'position_net')
        elif self.train_mode == 'lora':
            unet = self.unet
            grounding_downsampler = functools.partial(self.drivedreamer, 'grounding_downsampler')
            position_net = functools.partial(self.drivedreamer, 'position_net')
        else:
            assert False
        # convert images to latent space
        image = batch_dict['image'].to(self.vae.dtype)
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = latents.to(self.dtype)
        # sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        # sample a random timestep for each image
        # add noise to the model input according to the noise magnitude at each timestep
        bsz = latents.shape[0] // self.num_frames // self.num_cams
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device)
        timesteps = timesteps.long()
        if self.num_frames * self.num_cams > 1:
            timesteps = timesteps.repeat_interleave(self.num_frames * self.num_cams)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        if self.num_frames * self.num_cams > 1:
            timesteps = timesteps.reshape(bsz, -1)[:, 0]
        # add grounding downsampler latents
        grounding_downsampler_input = batch_dict.get('grounding_downsampler_input', None)
        if grounding_downsampler_input is not None:
            with self.accelerator.autocast():
                grounding_downsampler_input = grounding_downsampler_input.to(self.dtype)
                grounding_downsampler_latents = grounding_downsampler(grounding_downsampler_input).to(self.dtype)
                noisy_latents = torch.cat((noisy_latents, grounding_downsampler_latents), dim=1)
        # predict position_net
        position_net_input = dt_utils.to_dtype(batch_dict['position_net_input'], self.dtype)
        with self.accelerator.autocast():
            cross_attention_kwargs = dict(gligen=dict(objs=position_net(**position_net_input).to(self.dtype)))
        # predict the noise residual
        prompt_embeds = batch_dict['prompt_embeds'].to(self.dtype)
        with self.accelerator.autocast():
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                # added_cond_kwargs=dt_utils.to_dtype(batch_dict.get('added_cond_kwargs', None), self.dtype),
            ).sample
        # compute loss
        model_pred = model_pred.reshape(noise.shape)
        target = noise
        loss = F.mse_loss(model_pred.float(), target.float(), reduction='mean')
        return loss
