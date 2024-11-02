import torch

from .base_structure import BaseStructure


class Image(BaseStructure):
    def __init__(self, tensor):
        super(Image, self).__init__(tensor)
        assert self.tensor.ndim == 3
        self.tensor = self.tensor.to(torch.float32)

    @property
    def width(self):
        return self.tensor.shape[1]

    @property
    def height(self):
        return self.tensor.shape[0]
