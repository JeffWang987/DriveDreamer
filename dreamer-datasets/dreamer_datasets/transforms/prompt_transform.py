import html
import os
import re
import urllib.parse as ul

import ftfy
import torch

# from bs4 import BeautifulSoup
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5Tokenizer


class PromptTransform:
    def __init__(self, mode, model_path, device=None, dtype=None):
        if dtype is None:
            if device is not None and 'cuda' in device:
                dtype = torch.float16
            else:
                dtype = torch.float32
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.mode = mode
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.tokenizers = None
        self.text_encoders = None

    def load_model(self):
        if self.text_encoders is None:
            self.tokenizers = load_tokenizers(self.mode, self.model_path)
            self.text_encoders = load_text_encoders(self.mode, self.model_path, self.device, self.dtype)

    @torch.no_grad()
    def __call__(self, prompt, clean=False, max_length=None, with_attention_mask=False):
        self.load_model()
        if clean:
            prompt = clean_prompt(prompt)
            prompt = clean_prompt(prompt)
        prompt_ids, prompt_masks = forward_tokenizers(self.tokenizers, prompt, max_length)
        if not with_attention_mask:
            prompt_masks = None
        prompt_embeds = forward_text_encoders(self.text_encoders, self.mode, prompt_ids, prompt_masks)
        if with_attention_mask and len(prompt_masks) == 1:
            prompt_masks = prompt_masks[0]
        return prompt_embeds, prompt_masks


class PromptTokenizerTransform:
    def __init__(self, mode, model_path):
        self.mode = mode
        self.model_path = model_path
        self.tokenizers = None

    def load_model(self):
        if self.tokenizers is None:
            self.tokenizers = load_tokenizers(self.mode, self.model_path)

    def __call__(self, prompt, clean=False, max_length=None):
        self.load_model()
        if clean:
            prompt = clean_prompt(prompt)
            prompt = clean_prompt(prompt)
        prompt_ids, prompt_masks = forward_tokenizers(self.tokenizers, prompt, max_length)
        if len(prompt_ids) == 1:
            return prompt_ids[0], prompt_masks[0]
        else:
            return prompt_ids, prompt_masks


class PromptEncoderTransform:
    def __init__(self, mode, model_path, device=None, dtype=None):
        if dtype is None:
            if device is not None and 'cuda' in device:
                dtype = torch.float16
            else:
                dtype = torch.float32
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.mode = mode
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.text_encoders = None

    def load_model(self):
        if self.text_encoders is None:
            self.text_encoders = load_text_encoders(self.mode, self.model_path, self.device, self.dtype)

    @torch.no_grad()
    def __call__(self, prompt_ids, prompt_masks=None):
        self.load_model()
        prompt_embeds = forward_text_encoders(self.text_encoders, self.mode, prompt_ids, prompt_masks)
        return prompt_embeds


def load_tokenizers(mode, model_path):
    tokenizer_model_paths = []
    for tokenizer_name in ['tokenizer', 'tokenizer_2']:
        tokenizer_model_path = os.path.join(model_path, tokenizer_name)
        if os.path.exists(tokenizer_model_path):
            tokenizer_model_paths.append(tokenizer_model_path)
    if len(tokenizer_model_paths) == 0:
        tokenizer_model_paths = [model_path]
    if 'clip' in mode:
        tokenizers = [CLIPTokenizer.from_pretrained(_) for _ in tokenizer_model_paths]
    elif 't5' in mode:
        tokenizers = [T5Tokenizer.from_pretrained(_) for _ in tokenizer_model_paths]
    else:
        assert False
    return tokenizers


def load_text_encoders(mode, model_path, device, dtype):
    text_encoder_model_paths = []
    for text_encoder_name in ['text_encoder', 'text_encoder_2']:
        text_encoder_model_path = os.path.join(model_path, text_encoder_name)
        if os.path.exists(text_encoder_model_path):
            text_encoder_model_paths.append(text_encoder_model_path)
    if len(text_encoder_model_paths) == 0:
        text_encoder_model_paths = [model_path]
    if mode == 'clip_text':
        text_encoders = [CLIPTextModel.from_pretrained(text_encoder_model_paths[0], torch_dtype=dtype)]
    elif mode == 'clip_text_proj':
        text_encoders = [CLIPTextModelWithProjection.from_pretrained(text_encoder_model_paths[0], torch_dtype=dtype)]
    elif mode == 'clip_text_and_proj':
        text_encoders = [
            CLIPTextModel.from_pretrained(text_encoder_model_paths[0], torch_dtype=dtype),
            CLIPTextModelWithProjection.from_pretrained(text_encoder_model_paths[1], torch_dtype=dtype),
        ]
    elif mode == 't5':
        text_encoders = [T5EncoderModel.from_pretrained(text_encoder_model_paths[0], torch_dtype=dtype)]
    else:
        assert False
    for text_encoder in text_encoders:
        text_encoder.requires_grad_(False)
        if device is not None:
            text_encoder.to(device)
    return text_encoders


def forward_tokenizers(tokenizers, prompt, max_length=None):
    prompt_ids = []
    prompt_masks = []
    for tokenizer in tokenizers:
        max_length_i = max_length or tokenizer.model_max_length
        text_inputs = tokenizer(
            prompt,
            padding='max_length',
            max_length=max_length_i,
            truncation=True,
            return_tensors='pt',
        )
        input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        prompt_ids.append(input_ids)
        prompt_masks.append(attention_mask)
    return prompt_ids, prompt_masks


def forward_text_encoders(text_encoders, mode, prompt_ids, prompt_masks=None):
    device = text_encoders[0].device
    if mode in ('clip_text', 'clip_text_proj', 't5'):
        if isinstance(prompt_ids, list):
            prompt_ids = prompt_ids[0].to(device)
        else:
            prompt_ids = prompt_ids.to(device)
        if prompt_masks is not None:
            if isinstance(prompt_masks, list):
                prompt_masks = prompt_masks[0].to(device)
            else:
                prompt_masks = prompt_masks.to(device)
        with torch.no_grad():
            prompt_embeds = text_encoders[0](prompt_ids, attention_mask=prompt_masks)
        if mode in ('clip_text', 't5'):
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds = prompt_embeds[0].unsqueeze(1)

    elif mode == 'clip_text_and_proj':
        prompt_embeds_list = []
        for i, text_encoder in enumerate(text_encoders):
            prompt_ids_i = prompt_ids[i].to(device)
            prompt_masks_i = prompt_masks[i].to(device) if prompt_masks is not None else None
            with torch.no_grad():
                prompt_embeds = text_encoder(prompt_ids_i, attention_mask=prompt_masks_i, output_hidden_states=True)
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds = prompt_embeds.hidden_states[-2]
            prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        prompt_embeds = (prompt_embeds, pooled_prompt_embeds)

    else:
        assert False
    return prompt_embeds


def truncate_prompt(prompt_embeds, prompt_masks):
    assert len(prompt_embeds) == len(prompt_masks)
    new_prompt_embeds = []
    for i in range(len(prompt_embeds)):
        keep_index = prompt_masks[i].sum().item()
        prompt_embed = prompt_embeds[i][:keep_index]
        new_prompt_embeds.append(prompt_embed)
    return new_prompt_embeds[0] if len(new_prompt_embeds) == 1 else new_prompt_embeds


def pad_prompt(prompt_embeds, max_length, prompt_masks=None):
    cur_length = prompt_embeds.shape[0]
    assert cur_length <= max_length
    if prompt_masks is None:
        prompt_masks = torch.ones((cur_length,), device=prompt_embeds.device, dtype=torch.int64)
    if cur_length == max_length:
        return prompt_embeds, prompt_masks
    new_shape = list(prompt_embeds.shape)
    new_shape[0] = max_length
    new_prompt_embeds = torch.zeros(new_shape, device=prompt_embeds.device, dtype=prompt_embeds.dtype)
    new_prompt_masks = torch.zeros((max_length,), device=prompt_masks.device, dtype=prompt_masks.dtype)
    new_prompt_embeds[:cur_length] = prompt_embeds
    new_prompt_masks[:cur_length] = prompt_masks
    return new_prompt_embeds, new_prompt_masks


def clean_prompt(prompt):
    bad_punct_regex = re.compile(
        r'['
        + '#®•©™&@·º½¾¿¡§~'
        + r'\)'
        + r'\('
        + r'\]'
        + r'\['
        + r'\}'
        + r'\{'
        + r'\|'
        + '\\'
        + r'\/'
        + r'\*'
        + r']{1,}'
    )
    caption = str(prompt)
    caption = ul.unquote_plus(caption)
    caption = caption.strip().lower()
    caption = re.sub('<person>', 'person', caption)
    # urls:
    caption = re.sub(
        r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
        '',
        caption,
    )  # regex for urls
    caption = re.sub(
        r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
        '',
        caption,
    )  # regex for urls
    # html:
    # caption = BeautifulSoup(caption, features="html.parser").text

    # @<nickname>
    caption = re.sub(r'@[\w\d]+\b', '', caption)

    # 31C0—31EF CJK Strokes
    # 31F0—31FF Katakana Phonetic Extensions
    # 3200—32FF Enclosed CJK Letters and Months
    # 3300—33FF CJK Compatibility
    # 3400—4DBF CJK Unified Ideographs Extension A
    # 4DC0—4DFF Yijing Hexagram Symbols
    # 4E00—9FFF CJK Unified Ideographs
    caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
    caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
    caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
    caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
    caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
    caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
    caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)
    #######################################################

    # все виды тире / all types of dash --> "-"
    caption = re.sub(
        r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',  # noqa
        '-',
        caption,
    )

    # кавычки к одному стандарту
    caption = re.sub(r'[`´«»“”¨]', '"', caption)
    caption = re.sub(r'[‘’]', "'", caption)

    # &quot;
    caption = re.sub(r'&quot;?', '', caption)
    # &amp
    caption = re.sub(r'&amp', '', caption)

    # ip adresses:
    caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)

    # article ids:
    caption = re.sub(r'\d:\d\d\s+$', '', caption)

    # \n
    caption = re.sub(r'\\n', ' ', caption)

    # "#123"
    caption = re.sub(r'#\d{1,3}\b', '', caption)
    # "#12345.."
    caption = re.sub(r'#\d{5,}\b', '', caption)
    # "123456.."
    caption = re.sub(r'\b\d{6,}\b', '', caption)
    # filenames:
    caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)

    #
    caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
    caption = re.sub(r'[\.]{2,}', r' ', caption)  # """AUSVERKAUFT"""

    caption = re.sub(bad_punct_regex, r' ', caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
    caption = re.sub(r'\s+\.\s+', r' ', caption)  # " . "

    # this-is-my-cute-cat / this_is_my_cute_cat
    regex2 = re.compile(r'(?:\-|\_)')
    if len(re.findall(regex2, caption)) > 3:
        caption = re.sub(regex2, ' ', caption)

    caption = ftfy.fix_text(caption)
    caption = html.unescape(html.unescape(caption))

    caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
    caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
    caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231

    caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
    caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
    caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
    caption = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
    caption = re.sub(r'\bpage\s+\d+\b', '', caption)

    caption = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)  # j2d1a2a...

    caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

    caption = re.sub(r'\b\s+\:\s+', r': ', caption)
    caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
    caption = re.sub(r'\s+', ' ', caption)

    caption.strip()

    caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r'\1', caption)
    caption = re.sub(r"^[\'\_,\-\:;]", r'', caption)
    caption = re.sub(r"[\'\_,\-\:\-\+]$", r'', caption)
    caption = re.sub(r'^\.\S+$', '', caption)

    return caption.strip()
