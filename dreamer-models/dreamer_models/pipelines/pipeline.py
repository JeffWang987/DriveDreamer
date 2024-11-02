import os
from importlib import import_module

from torch.hub import get_dir

from .. import utils


class BasePipeline:
    def to(self, device):
        return self

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

