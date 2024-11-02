import torch

from .base_structure import BaseStructure
from .utils import points_utils


class Points(BaseStructure):
    def __init__(self, tensor):
        super(Points, self).__init__(tensor)
        assert self.tensor.ndim == 2 and self.tensor.shape[1] >= 2
        self.tensor = self.tensor.to(torch.float32)

    def points(self):
        return self.tensor.clone()

    def rotate(self, rot_mat_t):
        if not isinstance(rot_mat_t, torch.Tensor):
            rot_mat_t = self.tensor.new_tensor(rot_mat_t)
        self.tensor = points_utils.rotate_points(self.tensor, rot_mat_t)

    def flip(self, image_shape, direction='horizontal'):
        self.tensor = points_utils.flip_points(self.tensor, image_shape, direction)

    def clip(self, image_shape):
        self.tensor = points_utils.clip_points(self.tensor, image_shape)

    def crop(self, crop_range, mode='filter'):
        self.tensor, keep = points_utils.crop_points(self.tensor, crop_range, mode)
        return keep

    def scale(self, scale_hw, image_shape):
        self.tensor = points_utils.scale_points(self.tensor, scale_hw, image_shape)
