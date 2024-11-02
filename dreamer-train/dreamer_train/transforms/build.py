from ..registry import Registry, build_module

TRANSFORMS = Registry()


def build_transform(params_or_type, *args, **kwargs):
    return build_module(TRANSFORMS, params_or_type, *args, **kwargs)
