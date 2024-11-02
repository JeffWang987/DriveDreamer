import random

import numpy as np
import torch
from dreamer_datasets import PromptTokenizerTransform as _PromptTokenizerTransform
from dreamer_datasets import PromptTransform as _PromptTransform
from dreamer_datasets import utils as gd_utils
from dreamer_datasets.transforms.prompt_transform import pad_prompt
from dreamer_models import utils as gm_utils


class PromptTransform:
    def __init__(
        self,
        model_mode,
        model_path,
        prompt_mode='default',
        prompt_embeds_mode='default',
        default_prompt='',
        default_prompt_prob=0.0,
        clean_prompt=False,
        max_length=None,
        with_attention_mask=False,
        only_tokenizer=False,
        with_cache=False,
        cache_path=None,
        is_train=False,
    ):
        if only_tokenizer:
            self.transform = _PromptTokenizerTransform(model_mode, gm_utils.get_model_path(model_path))
        else:
            self.transform = _PromptTransform(model_mode, gm_utils.get_model_path(model_path))
        self.prompt_mode = prompt_mode
        self.prompt_embeds_mode = prompt_embeds_mode
        self.default_prompt = default_prompt
        self.default_prompt_prob = default_prompt_prob
        self.clean_prompt = clean_prompt
        self.max_length = max_length
        self.with_attention_mask = with_attention_mask
        self.only_tokenizer = only_tokenizer
        self.with_cache = with_cache
        self.is_train = is_train
        self.cache = dict()
        if cache_path is not None:
            self.cache = gd_utils.load_file(cache_path)

    def __call__(self, data_dict):
        if self.prompt_mode == 'default' or random.random() < self.default_prompt_prob:
            prompt = self.default_prompt
        elif self.prompt_mode == 'from_dict':
            prompt = data_dict['prompt']
            if prompt is not None:
                prompts = gd_utils.as_list(prompt)
                if self.is_train:
                    i = random.randint(0, len(prompts) - 1)
                else:
                    i = len(prompts) - 1
                prompt = prompts[i]
        else:
            assert False
        data_dict['prompt'] = prompt
        if not self.is_train:
            return data_dict
        if self.only_tokenizer:
            if prompt in self.cache:
                prompt_ids, prompt_masks = self.cache[prompt]
            elif self.prompt_embeds_mode == 'default':
                prompt_ids, prompt_masks = self.transform(
                    prompt,
                    clean=self.clean_prompt,
                    max_length=self.max_length,
                )
                if isinstance(prompt_ids, (list, tuple)):
                    prompt_ids = [p[0] for p in prompt_ids]
                else:
                    prompt_ids = prompt_ids[0]
                prompt_masks = prompt_masks[0]
                if not self.with_attention_mask:
                    prompt_masks = None
                if self.with_cache:
                    self.cache[prompt] = prompt_ids, prompt_masks
            else:
                assert False
            if isinstance(prompt_ids, (list, tuple)):
                data_dict['prompt_ids'] = prompt_ids[0]
                data_dict['prompt_masks'] = prompt_masks
                data_dict['added_cond_kwargs'] = {
                    'text_ids': prompt_ids[1],
                    'time_ids': self._get_add_time_ids(data_dict),
                }
            else:
                data_dict['prompt_ids'] = prompt_ids
                data_dict['prompt_masks'] = prompt_masks
        else:
            if prompt in self.cache:
                prompt_embeds, prompt_masks = self.cache[prompt]
            elif self.prompt_embeds_mode == 'default':
                prompt_embeds, prompt_masks = self.transform(
                    prompt,
                    clean=self.clean_prompt,
                    max_length=self.max_length,
                    with_attention_mask=self.with_attention_mask,
                )
                if isinstance(prompt_embeds, (list, tuple)):
                    prompt_embeds = [p[0] for p in prompt_embeds]
                else:
                    prompt_embeds = prompt_embeds[0]
                if prompt_masks is not None:
                    prompt_masks = prompt_masks[0]
                if self.with_cache:
                    self.cache[prompt] = prompt_embeds, prompt_masks
            elif self.prompt_embeds_mode == 'from_dict':
                prompt_embeds = torch.as_tensor(data_dict['prompt_embeds'])
                if 'prompt_masks' in data_dict:
                    prompt_masks = data_dict['prompt_masks']
                else:
                    prompt_masks = torch.ones((prompt_embeds.shape[0],), dtype=torch.int64)
                if self.max_length is not None:
                    prompt_embeds, prompt_masks = pad_prompt(
                        prompt_embeds=prompt_embeds,
                        prompt_masks=prompt_masks,
                        max_length=self.max_length,
                    )
                if not self.with_attention_mask:
                    prompt_masks = None
                if self.with_cache:
                    self.cache[prompt] = prompt_embeds, prompt_masks
            else:
                assert False
            if isinstance(prompt_embeds, (list, tuple)):
                data_dict['prompt_embeds'] = prompt_embeds[0]
                data_dict['prompt_masks'] = prompt_masks
                data_dict['added_cond_kwargs'] = {
                    'text_embeds': prompt_embeds[1],
                    'time_ids': self._get_add_time_ids(data_dict),
                }
            else:
                data_dict['prompt_embeds'] = prompt_embeds
                data_dict['prompt_masks'] = prompt_masks
        return data_dict

    def _get_add_time_ids(self, data_dict):
        if 'add_time_ids' in data_dict:
            add_time_ids = data_dict['add_time_ids']
        elif 'input_image' in data_dict:
            input_image = data_dict['input_image']
            crop_top_left = (0, 0)
            original_size = target_size = (input_image.shape[1], input_image.shape[2])
            add_time_ids = np.array(original_size + crop_top_left + target_size)
        else:
            assert False
        return add_time_ids
