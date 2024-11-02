import numpy as np
import torch

from ... import utils


class UNetMixin:
    def forward_unet(
        self,
        cur_step,
        latent_model_input,
        num_frames=None,
        chunk_size=None,
        unet=None,
        **kwargs,
    ):
        unet = unet or self.unet
        if num_frames is not None:
            if chunk_size == num_frames:
                noise_pred = unet(latent_model_input, **kwargs)[0]
            else:
                slices = uniform(cur_step, num_frames, chunk_size)
                if self.do_classifier_free_guidance:
                    for i in range(len(slices)):
                        slices[i] = torch.cat([slices[i], slices[i] + num_frames])
                noise_pred = utils.apply_slice(
                    unet,
                    latent_model_input,
                    **kwargs,
                    slices=slices,
                )[0]
        else:
            if self.do_classifier_free_guidance:
                total_size = len(latent_model_input) // 2
            else:
                total_size = len(latent_model_input)
            chunk_size = min(chunk_size or total_size, total_size)
            if chunk_size == total_size:
                noise_pred = unet(latent_model_input, **kwargs)[0]
            else:
                slices = list(torch.arange(total_size).chunk((total_size + 1) // chunk_size))
                if self.do_classifier_free_guidance:
                    for i in range(len(slices)):
                        slices[i] = torch.cat([slices[i], slices[i] + total_size])
                noise_pred = utils.apply_slice(
                    unet,
                    latent_model_input,
                    **kwargs,
                    slices=slices,
                )[0]
        return noise_pred

    def set_num_frames(self, num_frames):
        if num_frames is not None:
            for name, module in self.unet.named_modules():
                if hasattr(module, 'set_num_frames'):
                    module.set_num_frames(num_frames)


def ordered_halving(val):
    # get binary value, padded with 0s for 64 bits
    bin_str = f'{val:064b}'
    # flip binary value, padding included
    bin_flip = bin_str[::-1]
    # convert binary to int
    as_int = int(bin_flip, 2)
    # divide by 1 << 64, equivalent to 2**64, or 18446744073709551616,
    # or b10000000000000000000000000000000000000000000000000000000000000000 (1 with 64 zero's)
    final = as_int / (1 << 64)
    return final


def uniform(
    cur_step,
    num_frames,
    chunk_size,
    stride=1,
    overlap=-1,
    closed_loop=False,
):
    assert num_frames > chunk_size
    if overlap == -1:
        overlap = chunk_size // 4
    stride = min(stride, int(np.ceil(np.log2(num_frames / chunk_size))) + 1)
    context_steps = list(1 << np.arange(stride))
    contexts = []
    for context_step in context_steps:
        pad = int(round(num_frames * ordered_halving(cur_step)))
        start = int(ordered_halving(cur_step) * context_step) + pad
        end = num_frames + pad + (0 if closed_loop else -overlap)
        step = chunk_size * context_step - overlap
        for j in range(start, end, step):
            context = [e % num_frames for e in range(j, j + chunk_size * context_step, context_step)]
            contexts.append(torch.tensor(context, dtype=torch.int64))
    return contexts
