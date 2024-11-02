import os.path
import random

import numpy as np
import torch
from dreamer_datasets import CLIPTextTransform, CLIPTextWithProjectionTransform, CLIPTransform, boxes_utils, image_utils
from dreamer_models import utils as dm_utils
from dreamer_models.pipelines.vision.lane_detection import Lane2Hdmap
from torchvision import transforms

from .utils import generate_box_label, remove_none_label


class DriveDreamerTransform:
    def __init__(
        self,
        dst_size,
        mode,
        max_size=2048,
        multiple=64,
        dd_name=None,
        dd_resample=None,
        thickness=5,
        pos_name=None,
        max_objs=100,
        empty_prob=0.0,
        drop_text_or_image_prob=0.0,
        random_choice=None,
        prompt_name='sd',
        prompt_mode='default',
        default_prompt=None,
        with_cache=False,
        is_train=False,
    ):
        self.dd_name = dd_name
        self.pos_name = pos_name
        self.max_objs = max_objs
        self.is_train = is_train
        self.image_transform = ImageTransform(
            dst_size=dst_size,
            mode=mode,
            max_size=max_size,
            multiple=multiple,
            dd_name=dd_name,
            dd_resample=dd_resample,
            thickness=thickness,
        )
        self.pos_transform = PositionNetTransform(
            pos_name=pos_name,
            max_objs=max_objs,
            empty_prob=empty_prob,
            drop_text_or_image_prob=drop_text_or_image_prob,
            random_choice=random_choice,
            with_cache=with_cache,
            is_train=is_train,
        )
        self.prompt_transform = PromptTransform(
            prompt_name=prompt_name,
            prompt_mode=prompt_mode,
            default_prompt=default_prompt,
            with_cache=with_cache,
            is_train=is_train,
        )

    def __call__(self, data_dict):
        data_dict = self.image_transform(data_dict)
        data_dict = self.pos_transform(data_dict)
        data_dict = self.prompt_transform(data_dict)
        if self.is_train:
            new_data_dict = {
                'image': data_dict['input_image'],
                'grounding_downsampler_input': data_dict.get('input_dd_image', None),
                'position_net_input': data_dict['position_net_input'],
                'prompt_embeds': data_dict['prompt_embeds'],
                'added_cond_kwargs': data_dict.get('added_cond_kwargs', None),
            }
        else:
            new_data_dict = {
                'image': data_dict['image'],
                'prompt': data_dict['prompt'],
                'height': data_dict['input_image'].shape[1],
                'width': data_dict['input_image'].shape[2],
                'drivedreamer_image': data_dict.get('dd_image', None),
                'drivedreamer_boxes': data_dict.get('input_boxes', None),
                'drivedreamer_box_texts': data_dict.get('input_box_texts', None),
                'drivedreamer_box_images': data_dict.get('input_box_images', None),
                'drivedreamer_max_objs': self.max_objs,
            }
            # for visualization
            vis_dict = dict()
            if self.dd_name is not None:
                if self.dd_name not in data_dict:
                    vis_dict['lanes'] = data_dict['lanes']
                    vis_dict['lane_labels'] = data_dict['lane_labels']
                else:
                    vis_dict['dd_image'] = data_dict[self.dd_name]
            if self.pos_name is not None:
                if 'box' in self.pos_name:
                    vis_dict['boxes'] = data_dict['boxes']
                elif 'corner' in self.pos_name:
                    vis_dict['corners'] = data_dict['corners']
                else:
                    assert False
            new_data_dict['vis'] = vis_dict
        keys = list(new_data_dict.keys())
        for key in keys:
            if new_data_dict[key] is None:
                new_data_dict.pop(key)
        return new_data_dict


class ImageTransform:
    def __init__(
        self,
        dst_size,
        mode,
        max_size=2048,
        multiple=64,
        dd_name=None,
        dd_resample=None,
        thickness=5,
    ):
        if dd_name is not None and dd_resample is None:
            for key in ['hdmap']:
                if key in dd_name:
                    dd_resample = 'nearest'
        self.dst_size = dst_size
        self.mode = mode
        self.max_size = max_size
        self.multiple = multiple
        self.dd_name = dd_name
        self.dd_resample = dd_resample
        self.lane2hdmap = Lane2Hdmap(thickness)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __call__(self, data_dict):
        image = data_dict['image'].convert('RGB')
        input_image = image_utils.resize_image(
            image,
            self.dst_size,
            mode=self.mode,
            max_size=self.max_size,
            multiple=self.multiple,
        )
        if self.dd_name is not None:
            if self.dd_name not in data_dict:
                if 'hdmap' in self.dd_name:
                    lanes = data_dict['lanes']
                    lane_labels = data_dict.get('lane_labels', None)
                    dd_image = self.lane2hdmap(
                        lanes=lanes,
                        lane_labels=lane_labels,
                        image_size=image.size,
                        hdmap_size=input_image.size,
                    )
                else:
                    assert False
            else:
                dd_image = data_dict[self.dd_name]
            if dd_image.size != input_image.size:
                dd_image = dd_image.resize(input_image.size, image_utils.get_resample(self.dd_resample))
        else:
            dd_image = None
        input_image = self.transform(input_image)
        data_dict['input_image'] = input_image
        if dd_image is not None:
            input_dd_image = self.transform(dd_image)
            data_dict['dd_image'] = dd_image
            data_dict['input_dd_image'] = input_dd_image
        return data_dict


class PositionNetTransform:
    def __init__(
        self,
        pos_name=None,
        max_objs=100,
        empty_prob=0.0,
        drop_text_or_image_prob=0.0,
        random_choice=None,
        with_cache=False,
        is_train=False,
    ):
        self.pos_name = pos_name
        self.max_objs = max_objs
        self.empty_prob = empty_prob
        self.drop_text_or_image_prob = drop_text_or_image_prob
        self.is_train = is_train
        self.projection_matrix_path = os.path.join(dm_utils.get_model_dir(), 'dreamer_models/drivedreamer/projection_matrix')
        self.projection_matrix = None
        self.box_label_transform = BoxLabelTransform(
            pos_name=pos_name,
            random_choice=random_choice,
            with_cache=with_cache,
            is_train=is_train,
        )

    def choose_objs(self, data_dict):
        if 'box' in self.pos_name:
            boxes = data_dict['boxes']
            labels = data_dict['labels']
            ignores = data_dict.get('ignores', None)
        elif 'corner' in self.pos_name:
            image = data_dict['image']
            corners = data_dict['corners']
            labels = data_dict['labels3d']
            ignores = data_dict.get('ignores3d', None)
            boxes = boxes_utils.corners_to_boxes(corners)
            boxes = boxes_utils.clip_boxes(boxes, (image.height, image.width))
        else:
            assert False
        assert len(boxes) == len(labels)
        if ignores is not None:
            assert len(boxes) == len(ignores)
        if len(boxes) == 0:
            return []
        keeps = []
        for i in range(len(labels)):
            if labels[i] is None:
                continue
            if ignores is not None and ignores[i]:
                continue
            x1, y1, x2, y2 = boxes[i]
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            keeps.append(i)
        keeps = np.array(keeps, dtype=np.int64)
        if len(keeps) > self.max_objs:
            areas = []
            for i in keeps:
                x1, y1, x2, y2 = boxes[i]
                areas.append((x2 - x1) * (y2 - y1))
            indexes = np.array(areas).argsort()[::-1]
            keeps = keeps[indexes[: self.max_objs]]
        if self.is_train:
            keeps = np.random.permutation(keeps)
        return keeps

    def get_boxes(self, data_dict):
        image = data_dict['image']
        if 'box' in self.pos_name:
            boxes = data_dict['boxes'].copy()
            if boxes.ndim == 1:
                assert len(boxes) == 0
                boxes = np.zeros((0, 4), dtype=np.float32)
        elif 'corner' in self.pos_name:
            boxes = data_dict['corners'].copy()
            if boxes.ndim == 1:
                assert len(boxes) == 0
                boxes = np.zeros((0, 16), dtype=np.float32)
        else:
            assert False
        if boxes.ndim == 3:
            boxes = boxes.reshape((boxes.shape[0], boxes.shape[1] * boxes.shape[2]))
        assert boxes.ndim == 2
        boxes[:, ::2] /= image.width
        boxes[:, 1::2] /= image.height
        return boxes

    def get_labels(self, data_dict):
        if 'box' in self.pos_name:
            labels = data_dict['labels']
        elif 'corner' in self.pos_name:
            labels = data_dict['labels3d']
        else:
            assert False
        return labels

    def pad_data(self, data):
        data_shape = list(data.shape)
        assert data_shape[0] <= self.max_objs
        data_shape[0] = self.max_objs
        new_data = np.zeros(data_shape, dtype=data.dtype)
        new_data[: len(data)] = data
        return new_data

    def __call__(self, data_dict):
        if self.pos_name is not None:
            data_dict = self.box_label_transform(data_dict)
            keeps = self.choose_objs(data_dict)
            boxes = self.get_boxes(data_dict)[keeps]
            labels = self.get_labels(data_dict)
            labels = [labels[k] for k in keeps]
            label_embeds = data_dict['label_embeds'][keeps]
            masks = np.ones((len(keeps),), dtype=np.float32)
            if 'image' in self.pos_name:
                if self.is_train:
                    image_embeds = data_dict['box_image_embeds'][keeps]
                    label_masks = np.ones((len(keeps),), dtype=np.float32)
                    image_masks = np.ones((len(keeps),), dtype=np.float32)
                    if self.projection_matrix is None:
                        self.projection_matrix = torch.load(self.projection_matrix_path).numpy()
                    image_embeds = image_embeds @ self.projection_matrix
                    image_embeds = (image_embeds / np.linalg.norm(image_embeds)) * 28.7
                    if random.random() < self.empty_prob:
                        boxes[:] = 0
                        label_embeds[:] = 0
                        image_embeds[:] = 0
                        masks[:] = 0
                        label_masks[:] = 0
                        image_masks[:] = 0
                    elif self.drop_text_or_image_prob > 0:
                        for i in range(len(keeps)):
                            if random.random() < self.drop_text_or_image_prob:
                                idx = random.sample([0, 1], 1)[0]
                                if idx == 0:
                                    label_masks[i] = 0
                                else:
                                    image_masks[i] = 0
                    data_dict['position_net_input'] = {
                        'boxes': self.pad_data(boxes),
                        'masks': self.pad_data(masks),
                        'text_embeddings': self.pad_data(label_embeds),
                        'text_masks': self.pad_data(label_masks),
                        'image_embeddings': self.pad_data(image_embeds),
                        'image_masks': self.pad_data(image_masks),
                    }
                else:
                    box_images = data_dict['box_images'][keeps]
                    data_dict.update(
                        {
                            'input_boxes': boxes,
                            'input_box_texts': labels,
                            'input_box_images': box_images,
                        }
                    )
            else:
                if self.is_train:
                    if random.random() < self.empty_prob:
                        boxes[:] = 0
                        label_embeds[:] = 0
                        masks[:] = 0
                    data_dict['position_net_input'] = {
                        'boxes': self.pad_data(boxes),
                        'masks': self.pad_data(masks),
                        'positive_embeddings': self.pad_data(label_embeds),
                    }
                else:
                    data_dict.update(
                        {
                            'input_boxes': boxes,
                            'input_box_texts': labels,
                        }
                    )
        else:
            if self.is_train:
                data_dict['position_net_input'] = {
                    'image': data_dict['input_dd_image'],
                }
        return data_dict


class BoxLabelTransform:
    def __init__(self, pos_name, random_choice=None, with_cache=False, is_train=False):
        self.pos_name = pos_name
        self.random_choice = random_choice
        self.with_cache = with_cache
        self.is_train = is_train
        self.transform = CLIPTransform(model_path=dm_utils.get_model_path('openai/clip-vit-large-patch14'))
        self.cache = dict()

    def __call__(self, data_dict):
        if 'box' in self.pos_name:
            label_name = 'labels'
        elif 'corner' in self.pos_name:
            label_name = 'labels3d'
        else:
            assert False
        labels = []
        label_embeds = []
        for i, label in enumerate(data_dict[label_name]):
            label = remove_none_label(label)
            if len(label) == 0:
                self.transform.load_model()
                label_j = None
                label_embeds_j = np.zeros((self.transform.model.text_embed_dim,), dtype=np.float32)
            else:
                random_choice = self.random_choice if self.random_choice is not None else self.is_train
                label_j = generate_box_label(label, random_choice=random_choice)
                if label_j not in self.cache:
                    label_embeds_j = self.transform(label_j, text_w_proj=False)[0]
                    if self.with_cache:
                        self.cache[label_j] = label_embeds_j
                else:
                    label_embeds_j = self.cache[label_j]
            labels.append(label_j)
            label_embeds.append(label_embeds_j)
        if len(label_embeds) > 0:
            label_embeds = np.stack(label_embeds)
        else:
            self.transform.load_model()
            label_embeds = np.zeros((0, self.transform.model.text_embed_dim), dtype=np.float32)
        data_dict[label_name] = labels
        data_dict['label_embeds'] = label_embeds
        return data_dict


class PromptTransform:
    def __init__(
        self,
        prompt_name='sd',
        prompt_mode='default',
        default_prompt=None,
        with_cache=False,
        is_train=False,
    ):
        if prompt_mode == 'default':
            assert default_prompt is not None
        self.prompt_name = prompt_name
        self.prompt_mode = prompt_mode
        self.default_prompt = default_prompt
        self.with_cache = with_cache
        self.is_train = is_train
        if self.prompt_name == 'sd':
            self.transform = CLIPTextTransform(model_path=dm_utils.get_model_path('runwayml/stable-diffusion-v1-5'))
        elif prompt_name == 'sdxl':
            self.transform = CLIPTextWithProjectionTransform(
                model_path=dm_utils.get_model_path('stabilityai/stable-diffusion-xl-base-1.0')
            )
        else:
            assert False
        self.cache = dict()

    def extract_prompt_embeds(self, data_dict, prompt):
        if self.is_train:
            if prompt not in self.cache:
                prompt_embeds = self.transform(prompt, mode='before_pool')
                if self.with_cache:
                    self.cache[prompt] = prompt_embeds
            else:
                prompt_embeds = self.cache[prompt]
            if self.prompt_name == 'sd':
                data_dict['prompt_embeds'] = prompt_embeds[0]
            elif self.prompt_name == 'sdxl':
                data_dict['prompt_embeds'] = prompt_embeds[0][0]
                data_dict['added_cond_kwargs'] = {
                    'text_embeds': prompt_embeds[1][0],
                }
            else:
                assert False

    def __call__(self, data_dict):
        if self.prompt_mode == 'default':
            prompt = self.default_prompt
            add_prompt = []
            if 'scene_description' in data_dict:
                scene_description = data_dict['scene_description']
                if 'rain' in scene_description.lower():
                    add_prompt.append('rain')
                if 'night' in scene_description.lower():
                    add_prompt.append('night')

            if 'image_labels' in data_dict:
                add_prompt.append(data_dict['image_labels']['ISP'])
                add_prompt.extend(data_dict['image_labels']['view'])
            prompt = prompt[:-1]  # git rid of '.'
            prompt = ', '.join([prompt] + add_prompt) + '.'
            self.extract_prompt_embeds(data_dict, prompt)
        else:
            assert False
        if 'added_cond_kwargs' in data_dict:
            input_image = data_dict['input_image']
            crop_top_left = (0, 0)
            original_size = target_size = (input_image.shape[1], input_image.shape[2])
            add_time_ids = np.array(original_size + crop_top_left + target_size)
            data_dict['added_cond_kwargs']['time_ids'] = add_time_ids
        data_dict['prompt'] = prompt
        return data_dict
