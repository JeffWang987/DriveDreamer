import numpy as np
import torch


def _get_rot_mat_t(rot_sin, rot_cos):
    rot_mat_t = [rot_cos, rot_sin, -rot_sin, rot_cos]
    return rot_mat_t


def get_rot_mat_t(angle):
    # angle: (N,)
    if isinstance(angle, torch.Tensor):
        angle = angle.view(-1)
        rot_mat_t = _get_rot_mat_t(torch.sin(angle), torch.cos(angle))
        rot_mat_t = torch.cat(rot_mat_t, dim=0).reshape((2, 2, -1)).permute(2, 0, 1).contiguous()
    else:
        angle = np.array(angle)
        angle = angle.reshape((-1,))
        rot_mat_t = _get_rot_mat_t(np.sin(angle), np.cos(angle))
        rot_mat_t = np.concatenate(rot_mat_t, axis=0).reshape((2, 2, -1)).transpose(2, 0, 1)
    return rot_mat_t  # (N, 2, 2)


def rotate_points(points, rot_mat_t):
    # points: ([N], point_size, 2)
    # rot_mat_t: ([N], 2, 2) or ([N], 3, 2)
    is_tensor = isinstance(points, torch.Tensor)
    point_shape = points.shape
    rot_mat_t_shape = rot_mat_t.shape
    if point_shape[-1] == 2 and rot_mat_t_shape[-2] == 3:
        ones_shape = list(point_shape)[:-1] + [1]
        if is_tensor:
            points = torch.cat([points, points.new_ones(ones_shape)], dim=-1)
        else:
            points = np.concatenate([points, np.ones(ones_shape)], axis=-1)
    if len(point_shape) == 2:
        if len(rot_mat_t_shape) == 3:
            assert rot_mat_t_shape[0] == 1
            rot_mat_t = rot_mat_t[0]
        if is_tensor:
            return torch.einsum('ij,jk->ik', (points, rot_mat_t))
        else:
            return np.einsum('ij,jk->ik', points, rot_mat_t)
    elif len(point_shape) == 3:
        if is_tensor:
            return torch.einsum('aij,ajk->aik', points, rot_mat_t)
        else:
            return np.einsum('aij,ajk->aik', points, rot_mat_t)
    else:
        assert False


def flip_points(points, image_shape, direction='horizontal'):
    image_h, image_w = image_shape[:2]
    if direction == 'horizontal':
        points[:, 0] = image_w - 1 - points[:, 0]
    elif direction == 'vertical':
        points[:, 1] = image_h - 1 - points[:, 1]
    else:
        assert False
    return points


def clip_points(points, image_shape):
    image_h, image_w = image_shape[:2]
    if isinstance(points, torch.Tensor):
        points[:, 0] = torch.clamp(points[:, 0], min=0, max=image_w - 1)
        points[:, 1] = torch.clamp(points[:, 1], min=0, max=image_h - 1)
    else:
        points[:, 0] = np.clip(points[:, 0], a_min=0, a_max=image_w - 1)
        points[:, 1] = np.clip(points[:, 1], a_min=0, a_max=image_h - 1)
    return points


def crop_points(points, crop_range, mode='filter'):
    if isinstance(points, torch.Tensor):
        if mode == 'filter':
            keep = torch.logical_and(crop_range[:2] <= points, points <= crop_range[2:]).all(axis=1)
            points = points - crop_range[:2]
            keep = torch.nonzero(keep, as_tuple=False).view(-1)
            return points[keep], keep
        elif mode == 'clip':
            crop_range = crop_range.type_as(points)
            points = torch.min(torch.max(points, crop_range[0:2]), crop_range[2:4])
            points = points - crop_range[:2]
            return points, torch.arange(points.shape[0])
        else:
            assert False
    else:
        if mode == 'filter':
            keep = np.logical_and(crop_range[:2] <= points, points <= crop_range[2:]).all(axis=1)
            points = points - crop_range[:2]
            keep = np.where(keep == 1)[0]
            return points[keep], keep
        elif mode == 'clip':
            points = np.minimum(np.maximum(points, crop_range[0:2]), crop_range[2:4])
            points = points - crop_range[:2]
            return points, np.arange(points.shape[0])
        else:
            assert False


def scale_points(points, scale_hw, image_shape=None):
    scale_h, scale_w = scale_hw
    points[:, 0] *= scale_w
    points[:, 1] *= scale_h
    if image_shape is not None:
        points = clip_points(points, image_shape)
    return points
