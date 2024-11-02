import torch

from .base_structure import BaseStructure
from .utils import boxes_utils


class Boxes(BaseStructure):
    def __init__(self, tensor, offset=0):
        super(Boxes, self).__init__(tensor, offset=offset)
        assert self.tensor.shape[-1] == 4
        self.tensor = self.tensor.to(torch.float32)
        self.offset = offset

    @property
    def width(self):
        return boxes_utils.get_width(self.tensor, self.offset)

    @property
    def height(self):
        return boxes_utils.get_height(self.tensor, self.offset)

    @property
    def size(self):
        return boxes_utils.get_size(self.tensor, self.offset)

    @property
    def centers(self):
        return boxes_utils.get_centers(self.tensor)

    @property
    def dims(self):
        return boxes_utils.get_dims(self.tensor, self.offset)

    def corners(self):
        return boxes_utils.boxes_to_corners(self.tensor, self.offset)

    def boxes(self):
        return self.tensor.clone()

    def rotate(self, rot_mat_t):
        if not isinstance(rot_mat_t, torch.Tensor):
            rot_mat_t = self.tensor.new_tensor(rot_mat_t)
        self.tensor = boxes_utils.rotate_boxes(self.tensor, rot_mat_t, self.offset)

    def flip(self, image_shape, direction='horizontal'):
        self.tensor = boxes_utils.flip_boxes(self.tensor, image_shape, direction, self.offset)

    def clip(self, image_shape):
        self.tensor = boxes_utils.clip_boxes(self.tensor, image_shape, self.offset)

    def crop(self, crop_range, keep_outside_center=False, keep_outside_boxes=False):
        self.tensor, keep = boxes_utils.crop_boxes(crop_range, keep_outside_center, keep_outside_boxes)
        return keep

    def scale(self, scale_hw, image_shape):
        self.tensor = boxes_utils.scale_boxes(self.tensor, scale_hw, image_shape, self.offset)

    def overlaps(self, boxes, mode='iou', is_aligned=False, offset=0, eps=1e-6):
        if not isinstance(boxes, torch.Tensor):
            boxes = self.tensor.new_tensor(boxes)
        return boxes_utils.overlaps(self.tensor, boxes, mode=mode, is_aligned=is_aligned, offset=offset, eps=eps)
