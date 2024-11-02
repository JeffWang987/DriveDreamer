import copy
import os.path

import torch
from diffusers.models.attention import GatedSelfAttentionDense
from transformers import CLIPModel, CLIPProcessor

from ... import utils
from ...models import DriveDreamerModel
from .utils import add_control_model_name, repeat_data


class DriveDreamerMixin:
    def load_drivedreamer_model(self, pretrained_model_path, image_model_path=None, image_proj_path=None, **kwargs):
        kwargs.setdefault('torch_dtype', self.dtype)
        drivedreamer_kwargs = copy.deepcopy(kwargs)
        drivedreamer_kwargs.pop('variant', None)
        DriveDreamer = DriveDreamerModel.from_pretrained(pretrained_model_path, **drivedreamer_kwargs)
        DriveDreamer.to_unet(self.unet)
        self.DriveDreamer = DriveDreamer
        if image_model_path is None:
            image_model_path = utils.get_model_path('openai/clip-vit-large-patch14')
        self.DriveDreamer_clip_processor = CLIPProcessor.from_pretrained(image_model_path)
        self.DriveDreamer_clip_model = CLIPModel.from_pretrained(image_model_path, **drivedreamer_kwargs)
        if image_proj_path is None:
            image_proj_path = os.path.join(utils.get_model_dir(), 'dreamer_models/drivedreamer/projection_matrix')
        if os.path.exists(image_proj_path):
            self.DriveDreamer_image_proj = torch.load(image_proj_path).to(self.device)
        add_control_model_name(['DriveDreamer', 'DriveDreamer_clip_model'])

    def _enable_fuser(self, enabled=True):
        if hasattr(self, 'DriveDreamer'):
            for module in self.DriveDreamer.modules():
                if type(module) is GatedSelfAttentionDense:
                    module.enabled = enabled

    def forward_DriveDreamer(
        self,
        height,
        width,
        batch_size=1,
        num_images_per_prompt=1,
        num_frames=None,
        image=None,
        boxes=None,
        box_texts=None,
        box_images=None,
        max_objs=30,
        generator=None,
    ):
        if image is not None:
            image = self.image_processor.preprocess(image, height=height, width=width)
            image = image.to(device=self.device, dtype=self.DriveDreamer.dtype)
            latents = self.DriveDreamer('grounding_downsampler', image)
            latents = repeat_data(
                latents,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
            )
        else:
            latents = None
        if boxes is not None:
            boxes_maybe_video = [boxes] if boxes.dim() == 2 else boxes
            boxes_text_maybe_video = [box_texts] if boxes.dim() == 2 else box_texts
            box_images_maybe_video = [box_images] if boxes.dim() == 2 else box_images
            input_boxes_batch_list = []
            input_masks_batch_list = []
            text_embeddings_batch_list = []
            text_masks_batch_list = []
            image_embeddings_batch_list = []
            image_masks_batch_list = []
            for video_idx in range(len(boxes_maybe_video)):
                boxes = boxes_maybe_video[video_idx]
                box_texts = boxes_text_maybe_video[video_idx]
                if box_images_maybe_video is not None:
                    box_images = box_images_maybe_video[video_idx]
            
                feature_type = self.DriveDreamer.position_net.feature_type
                n_objs = min(len(box_texts), max_objs)
                boxes = boxes[:n_objs]
                input_boxes = torch.zeros((max_objs, boxes.shape[1]), device=self.device)
                input_masks = torch.zeros((max_objs,), device=self.device)
                text_embeddings = torch.zeros((max_objs, self.DriveDreamer_clip_model.text_embed_dim), device=self.device)
                text_masks = torch.zeros((max_objs,), device=self.device)
                image_embeddings = torch.zeros((max_objs, self.DriveDreamer_clip_model.projection_dim), device=self.device)
                image_masks = torch.zeros((max_objs,), device=self.device)
                if n_objs > 0:
                    boxes = torch.tensor(boxes, device=self.device)
                    input_boxes[:n_objs] = boxes[:n_objs]
                    input_masks[:n_objs] = 1
                    if box_texts is not None:
                        assert len(boxes) == len(box_texts)
                        for i in range(n_objs):
                            if box_texts[i] is None or box_texts[i] == '':
                                continue
                            text_inputs = self.DriveDreamer_clip_processor(
                                text=box_texts[i], padding=True, truncation=True, return_tensors='pt'
                            )
                            text_inputs = text_inputs.to(self.DriveDreamer_clip_model.device)
                            text_embedding = self.DriveDreamer_clip_model.text_model(**text_inputs)[1]
                            text_embeddings[i] = text_embedding[0]
                            text_masks[i] = 1
                    if box_images is not None:
                        assert len(boxes) == len(box_images) and feature_type == 'text_image'
                        for i in range(n_objs):
                            if box_images[i] is None:
                                continue
                            image_inputs = self.DriveDreamer_clip_processor(images=box_images[i], return_tensors='pt')
                            pixel_values = image_inputs.pixel_values.to(
                                self.DriveDreamer_clip_model.device, dtype=self.DriveDreamer_clip_model.dtype
                            )
                            image_embedding = self.DriveDreamer_clip_model.get_image_features(pixel_values)
                            image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)
                            image_embedding = image_embedding @ self.DriveDreamer_image_proj
                            image_embedding = (image_embedding / image_embedding.norm()) * 28.7
                            image_embeddings[i] = image_embedding[0]
                            image_masks[i] = 1
                input_boxes_batch_list.append(input_boxes)
                input_masks_batch_list.append(input_masks)
                text_embeddings_batch_list.append(text_embeddings)
                text_masks_batch_list.append(text_masks)
                image_embeddings_batch_list.append(image_embeddings)
                image_masks_batch_list.append(image_masks)
            if feature_type == 'text_only':
                pos_inputs = dict(
                    boxes=input_boxes_batch_list,
                    masks=input_masks_batch_list,
                    positive_embeddings=text_embeddings_batch_list,
                )
            elif feature_type == 'text_image':
                pos_inputs = dict(
                    boxes=input_boxes_batch_list,
                    masks=input_masks_batch_list,
                    text_embeddings=text_embeddings_batch_list,
                    text_masks=text_masks_batch_list,
                    image_embeddings=image_embeddings_batch_list,
                    image_masks=image_masks_batch_list,
                )
            else:
                assert False
            for key, val in pos_inputs.items():
                pos_inputs[key] = torch.stack(val, dim=0)
        elif image is not None:
            pos_inputs = dict(image=image)
        else:
            pos_inputs = None
        if pos_inputs is not None:
            for key, val in pos_inputs.items():
                val = val.to(dtype=self.DriveDreamer.dtype)
                val = repeat_data(
                    val,
                    batch_size=batch_size,
                    num_images_per_prompt=num_images_per_prompt,
                    num_frames=num_frames,
                )
                if self.do_classifier_free_guidance:
                    val = torch.cat([val] * 2)
                    val[: val.shape[0] // 2] = 0
                pos_inputs[key] = val
            objs = self.DriveDreamer('position_net', **pos_inputs)
        else:
            objs = None
        self._enable_fuser(True)
        outputs = dict(
            latents=latents,
            objs=objs,
        )
        return outputs

