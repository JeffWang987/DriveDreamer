import json
import os

import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from .utils import add_control_model_name, repeat_data


class PromptMixin:
    def load_prompt_model(self, pretrained_model_path, class_name=None, **kwargs):
        kwargs.setdefault('torch_dtype', self.dtype)
        tokenizer_model_path = os.path.join(pretrained_model_path, 'tokenizer')
        if not os.path.exists(tokenizer_model_path):
            tokenizer_model_path = pretrained_model_path
        text_encoder_model_path = os.path.join(pretrained_model_path, 'text_encoder')
        if not os.path.exists(text_encoder_model_path):
            text_encoder_model_path = pretrained_model_path
        if class_name is None:
            config_path = os.path.join(text_encoder_model_path, 'config.json')
            config = json.load(open(config_path, 'r'))
            class_name = config['_class_name']
        if class_name == 'CLIPTextModel':
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_model_path)
            self.text_encoder = CLIPTextModel.from_pretrained(text_encoder_model_path, **kwargs)
        elif class_name == 'CLIPTextModelWithProjection':
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_model_path)
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(text_encoder_model_path, **kwargs)
        elif class_name == 'T5EncoderModel':
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_model_path)
            self.text_encoder = T5EncoderModel.from_pretrained(text_encoder_model_path, **kwargs)
        else:
            assert False
        add_control_model_name('text_encoder')

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        batch_size=1,
        num_images_per_prompt=1,
        num_frames=None,
    ):
        device = self._execution_device
        text_inputs = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        if isinstance(self.text_encoder, CLIPTextModel):
            prompt_embeds = self.text_encoder(input_ids, attention_mask)
            prompt_embeds = prompt_embeds[0]
        elif isinstance(self.text_encoder, CLIPTextModelWithProjection):
            prompt_embeds = self.text_encoder(input_ids, attention_mask)
            prompt_embeds = prompt_embeds[0].unsqueeze(1)
        elif isinstance(self.text_encoder, T5EncoderModel):
            prompt_embeds = self.text_encoder(input_ids, attention_mask)
            prompt_embeds = truncate_prompt(prompt_embeds[0], attention_mask)[None]
        else:
            assert False
        prompt_embeds = repeat_data(
            prompt_embeds,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            num_frames=num_frames,
        )
        # get unconditional embeddings for classifier free guidance
        if self.do_classifier_free_guidance:
            if negative_prompt is None:
                uncond_tokens = [''] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f'`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !='
                    f' {type(prompt)}.'
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt] * batch_size
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f'`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:'
                    f' {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches'
                    ' the batch size of `prompt`.'
                )
            else:
                uncond_tokens = negative_prompt
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding='max_length',
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt',
            )
            input_ids = uncond_input.input_ids.to(device)
            attention_mask = uncond_input.attention_mask.to(device)
            if isinstance(self.text_encoder, CLIPTextModel):
                negative_prompt_embeds = self.text_encoder(input_ids, attention_mask)
                negative_prompt_embeds = negative_prompt_embeds[0]
            elif isinstance(self.text_encoder, CLIPTextModelWithProjection):
                negative_prompt_embeds = self.text_encoder(input_ids, attention_mask)
                negative_prompt_embeds = negative_prompt_embeds[0].unsqueeze(1)
            elif isinstance(self.text_encoder, T5EncoderModel):
                negative_prompt_embeds = self.text_encoder(input_ids, attention_mask)
                negative_prompt_embeds = negative_prompt_embeds[0][:, : prompt_embeds.shape[1]]
            else:
                assert False
            negative_prompt_embeds = repeat_data(
                negative_prompt_embeds,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                num_frames=num_frames,
            )
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds


def truncate_prompt(prompt_embeds, prompt_masks):
    assert len(prompt_embeds) == len(prompt_masks)
    new_prompt_embeds = []
    for i in range(len(prompt_embeds)):
        keep_index = prompt_masks[i].sum().item()
        prompt_embed = prompt_embeds[i, :keep_index]
        new_prompt_embeds.append(prompt_embed)
    return new_prompt_embeds[0] if len(new_prompt_embeds) == 1 else new_prompt_embeds
