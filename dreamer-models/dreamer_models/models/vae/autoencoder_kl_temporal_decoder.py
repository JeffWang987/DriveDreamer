from typing import Tuple, Union

import torch
from diffusers.models import AutoencoderKLTemporalDecoder as _AutoencoderKLTemporalDecoder
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils.accelerate_utils import apply_forward_hook


class AutoencoderKLTemporalDecoder(_AutoencoderKLTemporalDecoder):
    def enable_slicing(self):
        r"""Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several steps.
        This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self):
        r"""Disable sliced VAE decoding.

        If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    @apply_forward_hook
    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """Encode a batch of images into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        use_slicing = getattr(self, 'use_slicing', False)
        if use_slicing and x.shape[0] > 1:
            encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self.encoder(x)

        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(
        self,
        z: torch.FloatTensor,
        num_frames: int = 1,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """Decode a batch of images.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        batch_size = z.shape[0] // num_frames
        use_slicing = getattr(self, 'use_slicing', False)
        if use_slicing and batch_size > 1:
            image_only_indicator = torch.zeros(1, num_frames, dtype=z.dtype, device=z.device)
            decodeds = []
            for i in range(batch_size):
                z_slice = z[i * num_frames : (i + 1) * num_frames]
                decoded = self.decoder(z_slice, num_frames=num_frames, image_only_indicator=image_only_indicator)
                decodeds.append(decoded)
            decoded = torch.cat(decodeds)
        else:
            image_only_indicator = torch.zeros(batch_size, num_frames, dtype=z.dtype, device=z.device)
            decoded = self.decoder(z, num_frames=num_frames, image_only_indicator=image_only_indicator)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
