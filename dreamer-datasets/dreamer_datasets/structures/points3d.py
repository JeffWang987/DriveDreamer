import torch

from .base_structure import BaseStructure
from .mode3d import Mode3D
from .utils import boxes3d_utils, points3d_utils


class Points3D(BaseStructure):
    def __init__(self, tensor, mode=None):
        super(Points3D, self).__init__(tensor, mode=mode)
        assert self.tensor.ndim == 2 and self.tensor.shape[1] >= 3
        self.tensor = self.tensor.to(torch.float32)
        params = Mode3D.get_params(mode)
        self.mode = mode
        self.rot_axis = params['rot_axis']
        self.flip_axis = params['flip_axis']
        self.bev_axis = params['bev_axis']

    def convert_to(self, dst_mode, rot_mat=None):
        return points3d_utils.convert_points3d(self, src_mode=self.mode, dst_mode=dst_mode, rot_mat=rot_mat)

    def points(self):
        return self.tensor.clone()

    def points3d(self):
        return self.tensor.clone()

    def points2d(self, cam_intrinsic):
        """torch.Tensor: the shape of cam_intrinsic is (3, 4) or (3, 3)."""
        if not isinstance(cam_intrinsic, torch.Tensor):
            cam_intrinsic = self.tensor.new_tensor(cam_intrinsic)
        return points3d_utils.points3d_to_points2d(self.tensor, cam_intrinsic)

    def in_range(self, point_range):
        return points3d_utils.points3d_in_range(self.tensor, point_range)

    def in_range_bev(self, point_range):
        return points3d_utils.points3d_in_range_bev(self.tensor, point_range, self.bev_axis)

    def get_rot_mat_t(self, angle):
        if not isinstance(angle, torch.Tensor):
            angle = self.tensor.new_tensor(angle)
        return points3d_utils.get_rot_mat_t(angle, self.rot_axis)

    def rotate(self, angle=None, rot_mat_t=None):
        if rot_mat_t is None:
            rot_mat_t = self.get_rot_mat_t(angle)
        if not isinstance(rot_mat_t, torch.Tensor):
            rot_mat_t = self.tensor.new_tensor(rot_mat_t)
        self.tensor[:, :3] = points3d_utils.rotate_points3d(self.tensor[:, :3], rot_mat_t)

    def flip(self, bev_direction='horizontal'):
        self.tensor = points3d_utils.flip_points3d(self.tensor, self.flip_axis, bev_direction)

    def translate(self, trans_vector):
        if not isinstance(trans_vector, torch.Tensor):
            trans_vector = self.tensor.new_tensor(trans_vector)
        self.tensor = points3d_utils.translate_points3d(self.tensor, trans_vector)

    def scale(self, scale_factor):
        self.tensor = points3d_utils.scale_points3d(self.tensor, scale_factor)

    def points3d_in_surfaces3d(self, surfaces3d):
        points = self.tensor[:, :3]
        return points3d_utils.points3d_in_surfaces3d(points, surfaces3d)

    def points3d_in_boxes3d(self, boxes3d):
        points3d = self.tensor[:, :3]
        if not isinstance(boxes3d, torch.Tensor):
            boxes3d = self.tensor.new_tensor(boxes3d)
        return boxes3d_utils.points3d_in_boxes3d(points3d, boxes3d, self.rot_axis)
