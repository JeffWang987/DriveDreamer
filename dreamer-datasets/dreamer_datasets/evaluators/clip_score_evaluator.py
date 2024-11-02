import logging

import torch
from accelerate import PartialState
from transformers import CLIPModel, CLIPProcessor


class CLIPScoreEvaluator:
    def __init__(
        self,
        model_path,
        device=None,
        dtype=None,
        local_files_only=True,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.local_files_only = local_files_only
        self.processor = None
        self.model = None

    def load_model(self):
        if self.model is None:
            processor = CLIPProcessor.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only,
            )
            model = CLIPModel.from_pretrained(
                self.model_path,
                local_files_only=self.local_files_only,
            )
            model.eval()
            if self.device is not None:
                model.to(self.device)
            if self.dtype is not None:
                model.to(self.dtype)
            self.processor = processor
            self.model = model

    def __call__(self, results):
        if PartialState().is_main_process:
            self.load_model()
            clip_score_list = []
            for i, result in enumerate(results):
                image = result['image']
                prompt = result['prompt']
                clip_score = calculate_clip_score(prompt, image, self.processor, self.model)
                clip_score_list.append(clip_score)
            clip_score = sum(clip_score_list) / len(clip_score_list)
            logging.info('CLIP Score: {}'.format(clip_score))
            return dict(clip_score=clip_score)


def calculate_clip_score(prompt, image, processor, model):
    inputs = processor(
        text=prompt,
        images=image,
        padding=True,
        truncation=True,
        return_tensors='pt',
    )
    input_ids = inputs.input_ids
    pixel_values = inputs.pixel_values
    input_ids = input_ids.to(model.device)
    pixel_values = pixel_values.to(model.device, dtype=model.dtype)
    with torch.no_grad():
        outputs = model(input_ids, pixel_values)
        clip_score = outputs.logits_per_image.reshape(-1).item()
    return clip_score
