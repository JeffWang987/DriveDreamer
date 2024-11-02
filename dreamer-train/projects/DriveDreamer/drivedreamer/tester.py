import os

import numpy as np
import torch
import imageio

from dreamer_datasets import DefaultCollator, FIDEvaluator, ImageVisualizer, image_utils, load_dataset, DefaultSampler
from dreamer_datasets import utils as dd_utils
from dreamer_models import StableDiffusionControlPipeline
from dreamer_models import utils as dm_utils

from dreamer_train import Tester
from .transforms import DriveDreamerTransform
from .samplers import NuscVideoSampler


class DriveDreamerTester(Tester):
    def get_dataloaders(self, data_config):
        dataset = load_dataset(data_config.data_or_config)
        batch_size = data_config.batch_size_per_gpu
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
        pipeline_name = model_config.pipeline_name
        variant = 'fp16' if self.mixed_precision == 'fp16' else None
        self.num_frames = model_config.drivedreamer.get('num_frames', 1)
        self.is_video = self.num_frames > 1
        
        train_mode = model_config.get('train_mode', 'drivedreamer')
        weight_path = model_config.get('weight_path', None)
        with_ema = model_config.get('with_ema', False)
        ema_or_not = '_ema' if with_ema else ''
        if train_mode == 'drivedreamer':
            drivedreamer_model_path = os.path.join(self.get_checkpoint(weight_path), 'drivedreamer' + ema_or_not)
            unet3d_model_path = os.path.join(self.get_checkpoint(weight_path), 'unet' + ema_or_not)
            lora_model_path = None
        elif train_mode == 'lora':
            drivedreamer_model_path = os.path.join(self.get_checkpoint(model_config.checkpoint), 'drivedreamer' + ema_or_not)
            unet3d_model_path = os.path.join(self.get_checkpoint(model_config.checkpoint), 'unet' + ema_or_not)
            lora_model_path = os.path.join(self.get_checkpoint(weight_path), 'lora' + ema_or_not)
        else:
            assert False
            
        if pipeline_name == 'StableDiffusionControlPipeline':
            model = StableDiffusionControlPipeline.from_pretrained(
                dm_utils.get_model_path(model_config.pretrained),
                drivedreamer_model_path=drivedreamer_model_path,
                lora_model_path=lora_model_path,
                torch_dtype=self.dtype,
                variant=variant,
                local_files_only=True,
                safety_checker=None,
                unet_type=model_config.drivedreamer.get('unet_type', 'UNet2DConditionModel'),
                unet_from_2d_to_3d=model_config.drivedreamer.get('unet_from_2d_to_3d', False),
                num_frames=self.num_frames,
                unet3d_model_path=unet3d_model_path,
            )
        else:
            assert False

        model.to(self.device)
        return model

    def get_evaluators(self):
        # TODO add more
        evaluator = FIDEvaluator(
            dataset=self.dataloader.dataset,
            device=self.device,
            num_workers=self.dataloader.num_workers,
        )
        return evaluator

    def test(self):
        if self.is_main_process:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed)
            save_dir = self.kwargs.get('save_dir', None)
            if save_dir is not None:
                dd_utils.empty_dir(save_dir)
            evaluator = self.get_evaluators()
            results = []
            idx = 0
            max_count = self.kwargs.get('max_count', -1)
            for i, batch_dict in enumerate(self.dataloader):
                prompt = get_data(batch_dict, 'prompt', is_video=False)
                height = get_data(batch_dict, 'height', is_video=False)
                width = get_data(batch_dict, 'width', is_video=False)
                drivedreamer_max_objs = get_data(batch_dict, 'drivedreamer_max_objs', is_video=False)
                drivedreamer_image = get_data(batch_dict, 'drivedreamer_image', self.is_video)
                drivedreamer_boxes = get_data(batch_dict, 'drivedreamer_boxes', self.is_video)
                drivedreamer_box_texts = get_data(batch_dict, 'drivedreamer_box_texts', self.is_video)
                drivedreamer_box_images = get_data(batch_dict, 'drivedreamer_box_images', self.is_video)
                output_images = self.model(
                    prompt,
                    height=height,
                    width=width,
                    drivedreamer_image=drivedreamer_image,
                    drivedreamer_boxes=drivedreamer_boxes,
                    drivedreamer_box_texts=drivedreamer_box_texts,
                    drivedreamer_box_images=drivedreamer_box_images,
                    drivedreamer_max_objs=drivedreamer_max_objs,
                    generator=generator,
                    num_frames=self.num_frames,
                    guidance_scale=self.kwargs.get('guidance_scale', 7.5),
                ).images
                output_images = postprocess_images(output_images, batch_dict)
                if evaluator is not None:
                    for output_image in output_images:
                        results.append({'image': output_image})
                idx = save_images(output_images, batch_dict, idx=idx, save_dir=save_dir, is_video=self.is_video)
                if idx >= max_count > 0:
                    break
            if evaluator is not None:
                evaluator(results)
        self.accelerator.wait_for_everyone()


def get_data(batch_dict, key, is_video=False):
    if key in batch_dict:
        data = batch_dict[key]
        if isinstance(data, list):
            if isinstance(data[0], (list, tuple)):
                # assert len(data[0]) == 1
                if not is_video:
                    for i in range(len(data)):
                        data[i] = data[i][0]
            else:
                # assert len(data) == 1
                if not is_video:
                    data = data[0]
        else:
            # assert len(data) == 1
            if not is_video:
                data = data[0]
        return data
    else:
        return None


def postprocess_images(output_images, batch_dict):
    images = batch_dict['image']
    new_images = []
    for i, output_image in enumerate(output_images):
        output_image = output_image.resize(images[i].size)
        new_images.append(output_image)
    return new_images


def draw_images(output_images, batch_dict):
    vis_dict = batch_dict['vis']
    new_images = []
    for i, output_image in enumerate(output_images):
        output_image = ImageVisualizer(output_image)
        if 'lanes' in vis_dict:
            lanes = []
            lane_labels = []
            for lane, lane_label in zip(vis_dict['lanes'], vis_dict['lane_labels']):
                if isinstance(lane, torch.Tensor):
                    lane = lane.contiguous().cpu().numpy()
                lane = np.array(lane[i], dtype=np.float32).reshape((-1, 2)).copy()
                lanes.append(lane)
                lane_labels.append(lane_label[i])
            output_image.draw_lanes(lanes, lane_labels, thickness=2)
        if 'boxes' in vis_dict:
            boxes = vis_dict['boxes'][i].cpu().numpy()
            output_image.draw_boxes(boxes)
        elif 'corners' in vis_dict:
            corners = vis_dict['corners'][i].cpu().numpy()
            output_image.draw_corners(corners)
        new_images.append(output_image.get_image())
    return new_images


def save_images(output_images, batch_dict, idx, save_dir=None, each=True, show_height=None, is_video=False):
    if save_dir is None:
        return
    vis_images = draw_images(output_images, batch_dict)
    images = batch_dict['image']
    dd_images = batch_dict['vis'].get('dd_image', None)
    if is_video:
        video_frames = []
        for i in range(len(output_images)):
            cat_image = [images[i], vis_images[i]]
            if dd_images is not None:
                cat_image.insert(1, dd_images[i])
            cat_image = image_utils.concat_images(cat_image, pad=2)
            video_frames.append(cat_image)
        imageio.mimwrite(os.path.join(save_dir, '{:06d}.mp4'.format(idx)), video_frames, fps=12)
        return idx + 1
        
    else:
        if each:
            for i in range(len(output_images)):
                cat_images = [images[i], vis_images[i]]
                if dd_images is not None:
                    cat_images.insert(1, dd_images[i])
                cat_image = image_utils.concat_images(cat_images, pad=2)
                if show_height is not None:
                    cat_image = image_utils.resize_image(cat_image, dst_size=show_height, mode='height')
                cat_image.save(os.path.join(save_dir, '{:06d}.png'.format(idx + i)))
            return idx + len(output_images)
        else:
            cat_images = [
                image_utils.concat_images(images, pad=2),
                image_utils.concat_images(vis_images, pad=2),
            ]
            if dd_images is not None:
                cat_images.insert(1, image_utils.concat_images(dd_images, pad=2))
            cat_image = image_utils.concat_images(cat_images, direction='vertical', pad=2)
            if show_height is not None:
                cat_image = image_utils.resize_image(cat_image, dst_size=show_height, mode='height')
            cat_image.save(os.path.join(save_dir, '{:06d}.png'.format(idx)))
            return idx + 1
