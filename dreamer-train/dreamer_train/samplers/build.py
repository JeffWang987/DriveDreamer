from dreamer_datasets import DefaultSampler

from ..registry import Registry, build_module

SAMPLERS = Registry()


SAMPLERS.register_module(DefaultSampler)


def build_sampler(params_or_type, *args, **kwargs):
    return build_module(SAMPLERS, params_or_type, *args, **kwargs)
