import numpy as np
import torch

from . import boxes_utils, points3d_utils, points_utils


def get_volume(boxes3d, method=1, offset=0):
    if method == 1:
        width = boxes3d[:, 3]
        height = boxes3d[:, 4]
        depth = boxes3d[:, 5]
    elif method == 2:
        width = boxes3d[:, 3] - boxes3d[:, 0] + offset
        height = boxes3d[:, 4] - boxes3d[:, 1] + offset
        depth = boxes3d[:, 5] - boxes3d[:, 2] + offset
    else:
        assert False
    return width * height * depth


def get_pose_mat(boxes3d, rot_axis):
    centers3d, angles3d = boxes3d[:, :3], boxes3d[:, 6:9]
    rot_mat_t = points3d_utils.get_rot_mat_t(angles3d, rot_axis)
    if isinstance(boxes3d, torch.Tensor):
        rot_mat = rot_mat_t.permute(0, 2, 1).contiguous()  # (N, 3, 3)
        pose_mat = torch.cat([rot_mat, centers3d[:, :, None]], dim=-1)  # (N, 3, 4)
    else:
        rot_mat = rot_mat_t.transpose(0, 2, 1)  # (N, 3, 3)
        pose_mat = np.concatenate([rot_mat, centers3d[:, :, None]], axis=-1)  # (N, 3, 4)
    return pose_mat


def convert_boxes3d(boxes3d, src_mode, dst_mode, rot_mat=None):
    from ..boxes3d import Boxes3D
    from ..mode3d import Mode3D

    src_mode = Mode3D.get_mode(src_mode)
    dst_mode = Mode3D.get_mode(dst_mode)
    if src_mode == dst_mode:
        return boxes3d

    is_numpy = isinstance(boxes3d, np.ndarray)
    is_class = hasattr(boxes3d, 'tensor')
    if is_numpy:
        data = torch.from_numpy(boxes3d).clone()
    elif is_class:
        data = boxes3d.tensor.clone()
    else:
        data = boxes3d.clone()

    x_size, y_size, z_size = data[:, 3:4], data[:, 4:5], data[:, 5:6]
    x_rot, y_rot, z_rot = data[:, 6:7], data[:, 7:8], data[:, 8:9]

    if src_mode == Mode3D.LIDAR and dst_mode == Mode3D.CAMERA:
        if rot_mat is None:
            rot_mat = data.new_tensor([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
        xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
        z_rot = -z_rot - np.pi / 2
        z_rot = limit_period(z_rot, period=np.pi * 2)
        xyz_rot = torch.cat([x_rot, z_rot, y_rot], dim=-1)

    elif src_mode == Mode3D.CAMERA and dst_mode == Mode3D.LIDAR:
        if rot_mat is None:
            rot_mat = data.new_tensor([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
        xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
        y_rot = -y_rot - np.pi / 2
        y_rot = limit_period(y_rot, period=np.pi * 2)
        xyz_rot = torch.cat([x_rot, z_rot, y_rot], dim=-1)

    elif src_mode == Mode3D.DEPTH and dst_mode == Mode3D.CAMERA:
        if rot_mat is None:
            rot_mat = data.new_tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
        z_rot = -z_rot
        xyz_rot = torch.cat([x_rot, z_rot, y_rot], dim=-1)

    elif src_mode == Mode3D.CAMERA and dst_mode == Mode3D.DEPTH:
        if rot_mat is None:
            rot_mat = data.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        xyz_size = torch.cat([x_size, z_size, y_size], dim=-1)
        y_rot = -y_rot
        xyz_rot = torch.cat([x_rot, z_rot, y_rot], dim=-1)

    elif src_mode == Mode3D.LIDAR and dst_mode == Mode3D.DEPTH:
        if rot_mat is None:
            rot_mat = data.new_tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        xyz_size = torch.cat([x_size, y_size, z_size], dim=-1)
        z_rot = z_rot + np.pi / 2
        z_rot = limit_period(z_rot, period=np.pi * 2)
        xyz_rot = torch.cat([x_rot, y_rot, z_rot], dim=-1)

    elif src_mode == Mode3D.DEPTH and dst_mode == Mode3D.LIDAR:
        if rot_mat is None:
            rot_mat = data.new_tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        xyz_size = torch.cat([x_size, y_size, z_size], dim=-1)
        z_rot = z_rot - np.pi / 2
        z_rot = limit_period(z_rot, period=np.pi * 2)
        xyz_rot = torch.cat([x_rot, y_rot, z_rot], dim=-1)

    else:
        assert False

    if not isinstance(rot_mat, torch.Tensor):
        rot_mat = data.new_tensor(rot_mat)
    if rot_mat.size(1) == 4:
        extended_xyz = torch.cat([data[:, :3], data.new_ones(data.size(0), 1)], dim=-1)
        xyz = torch.matmul(extended_xyz, rot_mat.t())
    else:
        xyz = torch.matmul(data[:, :3], rot_mat.t())

    boxes3d = torch.cat([xyz[:, :3], xyz_size, xyz_rot, data[:, 9:]], dim=-1)

    if is_numpy:
        return boxes3d.numpy()
    elif is_class:
        return Boxes3D(boxes3d, mode=dst_mode)
    else:
        return boxes3d


def dims3d_to_norm_corners3d(dims3d, origin=(0.5, 0.5, 0.5)):
    # dims3d: (N, 3)
    norm_corners3d = np.unravel_index(np.arange(8), [2, 2, 2])
    norm_corners3d = np.stack(norm_corners3d, axis=1)  # (8, 3)
    if isinstance(dims3d, torch.Tensor):
        norm_corners3d = dims3d.new_tensor(norm_corners3d)
        origin = dims3d.new_tensor(origin)
    else:
        norm_corners3d = norm_corners3d.astype(dims3d.dtype)
        origin = np.array(origin, dtype=dims3d.dtype)
    norm_corners3d = norm_corners3d[[0, 1, 3, 2, 4, 5, 7, 6]]  # (8, 3)
    norm_corners3d = norm_corners3d - origin
    norm_corners3d = dims3d.reshape((-1, 1, 3)) * norm_corners3d.reshape((1, 8, 3))  # (N, 8, 3)
    return norm_corners3d


def corners3d_to_dims3d(corners3d):
    # corners3d: (N, 8, 3)
    minmax_boxes3d = corners3d_to_minmax_boxes3d(corners3d)
    dims3d = minmax_boxes3d[:, 3:] - minmax_boxes3d[:, :3]
    return dims3d


def corners3d_to_minmax_boxes3d(corners3d):
    # corners3d: (N, 8, 3)
    if isinstance(corners3d, torch.Tensor):
        minmax_boxes3d = corners3d.new_zeros((corners3d.shape[0], 6))
        minmax_boxes3d[:, :3] = corners3d.min(1)[0]
        minmax_boxes3d[:, 3:] = corners3d.max(1)[0]
    else:
        minmax_boxes3d = np.zeros((corners3d.shape[0], 6), dtype=corners3d.dtype)
        minmax_boxes3d[:, :3] = corners3d.min(1)
        minmax_boxes3d[:, 3:] = corners3d.max(1)
    return minmax_boxes3d


def corners3d_to_corners2d(corners3d, cam_intrinsic, ret_depth=False):
    # corners3d: (N, 8, 3)
    # cam_intrinsic: (3, 4) or (3, 3)
    corners_shape = corners3d.shape[:-1] + (1,)
    if isinstance(corners3d, torch.Tensor):
        if cam_intrinsic.shape[1] == 4:
            ones = corners3d.new_ones(corners_shape, dtype=corners3d.dtype)
            corners3d = torch.cat([corners3d, ones], dim=-1)  # (N, 8, 4)
        corners3d = torch.matmul(corners3d, cam_intrinsic.t())  # (N, 8, 3)
    else:
        if cam_intrinsic.shape[1] == 4:
            ones = np.ones(corners_shape, dtype=corners3d.dtype)
            corners3d = np.concatenate([corners3d, ones], axis=-1)  # (N, 8, 4)
        corners3d = np.dot(corners3d, cam_intrinsic.T)  # (N, 8, 3)
    depths = corners3d[..., 2:3]
    corners = corners3d[..., :2] / depths  # (N, 8, 2)
    if ret_depth:
        return corners, depths
    else:
        return corners


def corners2d_to_corners3d(corners, depths, cam_intrinsic):
    # corners: (N, 8, 2)
    # depths: (N, 8, 1) or (N, 8)
    # cam_intrinsic: (3, 4) or (3, 3)
    if corners.ndim == depths.ndim + 1:
        depths = depths[..., None]
    corners_shape = corners.shape[:-1] + (1,)
    if isinstance(corners, torch.Tensor):
        corners3d = torch.cat([corners * depths, depths], dim=1)
        if cam_intrinsic.shape[1] == 4:
            ones = corners3d.new_ones(corners_shape, dtype=corners3d.dtype)
            corners3d = torch.cat([corners3d, ones], dim=-1)  # (N, 8, 4)
        corners3d = torch.matmul(corners3d, torch.inverse(cam_intrinsic).t())  # (N, 8, 4)
    else:
        corners3d = np.concatenate([corners * depths, depths], axis=-1)
        if cam_intrinsic.shape[1] == 4:
            ones = np.ones(corners_shape, dtype=corners3d.dtype)
            corners3d = np.concatenate([corners3d, ones], axis=-1)  # (N, 8, 4)
        corners3d = np.dot(corners3d, np.linalg.inv(cam_intrinsic).T)  # (N, 8, 4)
    corners3d = corners3d[..., :3]
    return corners3d


def corners3d_to_surfaces3d(corners3d):
    # corners3d: (N, 8, 3)
    corner_indexes = [0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7]
    surfaces3d = [corners3d[:, idx] for idx in corner_indexes]
    if isinstance(corners3d, torch.Tensor):
        surfaces3d = torch.cat(surfaces3d, dim=0)
        surfaces3d = surfaces3d.view(6, 4, -1, 3)  # (6, 4, N, 3)
        surfaces3d = surfaces3d.permute((2, 0, 1, 3)).contiguous()  # (N, 6, 4, 3)
    else:
        surfaces3d = np.concatenate(surfaces3d, axis=0)
        surfaces3d = surfaces3d.reshape((6, 4, -1, 3))  # (6, 4, N, 3)
        surfaces3d = surfaces3d.transpose((2, 0, 1, 3))  # (N, 6, 4, 3)
    return surfaces3d


def boxes3d_to_corners3d(boxes3d, rot_axis):
    # boxes3d: (N, 9)
    centers3d, dims3d, angles3d = boxes3d[:, :3], boxes3d[:, 3:6], boxes3d[:, 6:9]
    norm_corners3d = dims3d_to_norm_corners3d(dims3d)  # (N, 8, 3)
    rot_mat_t = points3d_utils.get_rot_mat_t(angles3d, rot_axis)
    corners3d = points3d_utils.rotate_points3d(norm_corners3d, rot_mat_t)  # (N, 8, 3)
    corners3d += centers3d.reshape((-1, 1, 3))  # (N, 8, 3)
    return corners3d


def boxes3d_to_corners2d(boxes3d, rot_axis, cam_intrinsic):
    # boxes3d: (N, 9)
    corners3d = boxes3d_to_corners3d(boxes3d, rot_axis)  # (N, 8, 3)
    corners = corners3d_to_corners2d(corners3d, cam_intrinsic)  # (N, 8, 2)
    return corners


def boxes3d_to_minmax_boxes3d(boxes3d, rot_axis):
    # boxes3d: (N, 9)
    corners3d = boxes3d_to_corners3d(boxes3d, rot_axis)  # (N, 8, 3)
    minmax_boxes3d = corners3d_to_minmax_boxes3d(corners3d)  # (N, 6)
    return minmax_boxes3d


def boxes3d_to_bev_boxes2d(boxes3d, bev_axis):
    # boxes3d: (N, 9)
    if bev_axis[-1] < 0:
        bev_axis = list(bev_axis[:-1]) + [-bev_axis[-1]]
        bev_boxes = boxes3d[:, bev_axis]
        bev_boxes[:, -1] = -bev_boxes[:, -1]
    else:
        bev_boxes = boxes3d[:, bev_axis]
    return bev_boxes


def bev_boxes2d_to_nearest_bev_boxes2d(bev_boxes):
    # bev_boxes: (N, 5)
    assert bev_boxes.shape[1] == 5
    normed_rotations = limit_period(bev_boxes[:, 4], 0.5, np.pi)  # limit rad to [-pi / 2, pi / 2]
    if isinstance(bev_boxes, torch.Tensor):
        normed_rotations = torch.abs(normed_rotations)  # limit rad to [0, pi / 2]
        conditions = normed_rotations[:, None] < np.pi / 4
        dims = torch.where(conditions, bev_boxes[:, [2, 3]], bev_boxes[:, [3, 2]])
        centers = bev_boxes[:, :2]
        nearest_bev_boxes = torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)
    else:
        normed_rotations = np.abs(normed_rotations)  # limit rad to [0, pi / 2]
        conditions = normed_rotations[:, None] < np.pi / 4
        dims = np.where(conditions, bev_boxes[:, [2, 3]], bev_boxes[:, [3, 2]])
        centers = bev_boxes[:, :2]
        nearest_bev_boxes = np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)
    return nearest_bev_boxes


def boxes3d_to_nearest_bev_boxes2d(boxes3d, bev_axis):
    # boxes3d: (N, 9)
    bev_boxes = boxes3d_to_bev_boxes2d(boxes3d, bev_axis)
    nearest_bev_boxes = bev_boxes2d_to_nearest_bev_boxes2d(bev_boxes)
    return nearest_bev_boxes


def boxes3d_to_boxes2d(boxes3d, rot_axis, cam_intrinsic):
    # boxes3d: (N, 9)
    corners = boxes3d_to_corners2d(boxes3d, rot_axis, cam_intrinsic)  # (N, 8, 2)
    boxes = boxes_utils.corners_to_boxes(corners)  # (N, 4)
    return boxes


def boxes3d_to_surfaces3d(boxes3d, rot_axis):
    # boxes3d: (N, 9)
    corners3d = boxes3d_to_corners3d(boxes3d, rot_axis)  # (N, 8, 3)
    surfaces3d = corners3d_to_surfaces3d(corners3d)
    return surfaces3d


def bev_boxes2d_to_corners2d(bev_boxes):
    # bev_boxes: (N, 5)
    centers, dims, angles = bev_boxes[:, :2], bev_boxes[:, 2:4], bev_boxes[:, 4]
    norm_corners = boxes_utils.dims_to_norm_corners(dims)  # (N, 4, 2)
    rot_mat_t = points_utils.get_rot_mat_t(angles)
    corners = points_utils.rotate_points(norm_corners, rot_mat_t)  # (N, 4, 2)
    corners += centers.reshape((-1, 1, 2))  # (N, 4, 2)
    return corners


def corners3d_in_image(corners3d, cam_intrinsic, image_shape=None):
    # corners3d: (N, 8, 3)
    flag = corners3d[..., 2].mean(-1) > 0
    if image_shape is not None:
        corners = corners3d_to_corners2d(corners3d, cam_intrinsic)
        in_image = boxes_utils.corners_in_image(corners, image_shape)
        if isinstance(corners3d, torch.Tensor):
            flag = torch.logical_and(in_image, flag)
        else:
            flag = np.logical_and(in_image, flag)
    return flag


def boxes3d_in_image(boxes3d, rot_axis, cam_intrinsic, image_shape):
    # boxes3d: (N, 9)
    corners3d = boxes3d_to_corners3d(boxes3d, rot_axis)
    return corners3d_in_image(corners3d, cam_intrinsic, image_shape)


def rotate_boxes3d(boxes3d, rot_mat_t, angle, rot_axis):
    # boxes3d: (N, 9)
    boxes3d[:, :3] = points3d_utils.rotate_points3d(boxes3d[:, :3], rot_mat_t)
    if rot_axis >= 0:
        boxes3d[:, 6 + rot_axis] += angle
    else:
        boxes3d[:, 6:9] += angle
    return boxes3d


def flip_boxes3d(boxes3d, flip_axis, rot_axis, bev_direction='horizontal'):
    # boxes3d: (N, 9)
    if flip_axis[0] == 0:
        offset_h, offset_v = np.pi, 0
    elif flip_axis[0] == 1:
        offset_h, offset_v = 0, np.pi
    else:
        assert False
    if bev_direction == 'horizontal':
        boxes3d[:, flip_axis[0]] = -boxes3d[:, flip_axis[0]]
        boxes3d[:, 6 + rot_axis] = -boxes3d[:, 6 + rot_axis] + offset_h
    elif bev_direction == 'vertical':
        boxes3d[:, flip_axis[1]] = -boxes3d[:, flip_axis[1]]
        boxes3d[:, 6 + rot_axis] = -boxes3d[:, 6 + rot_axis] + offset_v
    else:
        assert False
    return boxes3d


def translate_boxes3d(boxes3d, trans_vector):
    boxes3d[:, :3] += trans_vector
    return boxes3d


def scale_boxes3d(boxes3d, scale_factor):
    boxes3d[:, :6] *= scale_factor
    return boxes3d


def crop_corners3d(corners3d, z_b=0.5):
    for i in range(len(corners3d)):
        back_cam_mask = corners3d[i, :, 2] < 0
        if back_cam_mask.sum() > 0:
            ori_back_pts = corners3d[i, back_cam_mask]
            to_be_adjusted_pts = corners3d[i, back_cam_mask].copy()
            to_be_adjusted_pts[:, 2] = z_b
            back_idx = np.arange(8)[back_cam_mask]
            opposite_idx = (back_idx + 4) % 8
            corespoding_pts = corners3d[i, opposite_idx]
            k_scale = (corespoding_pts[:, 2] - z_b) / (corespoding_pts[:, 2] - ori_back_pts[:, 2])  # 8
            to_be_adjusted_pts[:, :2] = corespoding_pts[:, :2] - k_scale[:, None] * (
                corespoding_pts[:, :2] - ori_back_pts[:, :2]
            )
            corners3d[i, back_cam_mask] = to_be_adjusted_pts
    return corners3d


def limit_period(val, offset=0.5, period=np.pi):
    # limit to: [-offset*period, period-offset*period]
    if isinstance(val, torch.Tensor):
        return val - torch.floor(val / period + offset) * period
    else:
        return val - np.floor(val / period + offset) * period


def limit_angles(boxes3d, rot_axis, offset=0.5, period=np.pi):
    ranges = range(6, 9) if rot_axis == -1 else [6 + rot_axis]
    for i in ranges:
        boxes3d[:, i] = limit_period(boxes3d[:, i], offset, period)
    return boxes3d


def points3d_in_boxes3d(points3d, boxes3d, rot_axis):
    # points3d: (num_points, 3)
    # boxes3d: (num_boxes, 9)
    # indices: (num_points, num_boxes)
    surfaces3d = boxes3d_to_surfaces3d(boxes3d, rot_axis)
    indices = points3d_utils.points3d_in_surfaces3d(points3d, surfaces3d)
    return indices


def overlaps(
    boxes3d_a,
    boxes3d_b,
    method,
    mode='iou',
    offset=0,
    eps=1e-6,
    rot_axis=None,
    bev_axis=None,
):
    # boxes3d_a: (n, 9)
    # boxes3d_b: (m, 9)
    # return: (n, m)

    if boxes3d_a.shape[0] * boxes3d_b.shape[0] == 0:
        if isinstance(boxes3d_a, torch.Tensor):
            return boxes3d_a.new_zeros((boxes3d_a.shape[0], boxes3d_b.shape[0]))
        else:
            return np.zeros((boxes3d_a.shape[0], boxes3d_b.shape[0]))
    if method.startswith('minmax'):
        if method == 'minmax_boxes':
            boxes3d_a = boxes3d_to_minmax_boxes3d(boxes3d_a, rot_axis)  # (n, 6)
            boxes3d_b = boxes3d_to_minmax_boxes3d(boxes3d_b, rot_axis)  # (m, 6)
        elif method == 'minmax_corners':
            corners3d_a = boxes3d_a.reshape((-1, 8, 3))
            corners3d_b = boxes3d_b.reshape((-1, 8, 3))
            boxes3d_a = corners3d_to_minmax_boxes3d(corners3d_a)  # (n, 6)
            boxes3d_b = corners3d_to_minmax_boxes3d(corners3d_b)  # (m, 6)
        else:
            assert False
        if isinstance(boxes3d_a, torch.Tensor):
            lt = torch.max(boxes3d_a[:, None, :3], boxes3d_b[None, :, :3])  # (n, m, 3)
            rb = torch.min(boxes3d_a[:, None, 3:6], boxes3d_b[None, :, 3:6])  # (n, m, 3)
            wh = (rb - lt + offset).clamp(min=0)  # (n, m, 3)
        else:
            lt = np.maximum(boxes3d_a[:, None, :3], boxes3d_b[None, :, :3])  # (n, m, 3)
            rb = np.minimum(boxes3d_a[:, None, 3:6], boxes3d_b[None, :, 3:6])  # (n, m, 3)
            wh = np.maximum(rb - lt + offset, 0)  # (n, m, 3)
        overlap = wh[:, :, 0] * wh[:, :, 1] * wh[:, :, 2]  # (n, m)
        if mode is None:
            return overlap
        if mode in ('iou', 'ioa'):
            volume_a = get_volume(boxes3d_a, method=2, offset=offset)
        if mode in ('iou', 'iob'):
            volume_b = get_volume(boxes3d_b, method=2, offset=offset)
        if mode == 'iou':  # intersection over union
            union = volume_a[:, None] + volume_b[None, :] - overlap  # (n, m)
        elif mode == 'ioa':  # intersection over a
            union = volume_a[:, None]  # (n, m)
        elif mode == 'iob':  # intersection over b
            union = volume_b[None, :]  # (n, m)
        else:
            assert False
        if isinstance(boxes3d_a, torch.Tensor):
            union = torch.max(union, union.new_tensor([eps]))
        else:
            union = np.maximum(union, eps)
        ios = overlap / union  # (n, m)
        return ios

    elif method.startswith('depth'):
        # compute overlap_d
        depth_axis = 3 - sum(bev_axis[:2])
        boxes3d_a_d1 = boxes3d_a[:, depth_axis] - boxes3d_a[:, depth_axis + 3] / 2
        boxes3d_a_d2 = boxes3d_a_d1 + boxes3d_a[:, depth_axis + 3]
        boxes3d_b_d1 = boxes3d_b[:, depth_axis] - boxes3d_b[:, depth_axis + 3] / 2
        boxes3d_b_d2 = boxes3d_b_d1 + boxes3d_b[:, depth_axis + 3]
        if isinstance(boxes3d_a, torch.Tensor):
            boxes3d_d1 = torch.max(boxes3d_a_d1[:, None], boxes3d_b_d1[None, :])
            boxes3d_d2 = torch.min(boxes3d_a_d2[:, None], boxes3d_b_d2[None, :])
            overlap_d = torch.clamp(boxes3d_d2 - boxes3d_d1 + offset, min=0)  # (n, m)
        else:
            boxes3d_d1 = np.maximum(boxes3d_a_d1[:, None], boxes3d_b_d1[None, :])
            boxes3d_d2 = np.minimum(boxes3d_a_d2[:, None], boxes3d_b_d2[None, :])
            overlap_d = np.maximum(boxes3d_d2 - boxes3d_d1 + offset, 0)  # (n, m)
        # compute overlap_bev
        if method == 'depth_nearest_bev':
            boxes_a_bev = boxes3d_to_nearest_bev_boxes2d(boxes3d_a, bev_axis)
            boxes_b_bev = boxes3d_to_nearest_bev_boxes2d(boxes3d_b, bev_axis)
            if isinstance(boxes3d_a, torch.Tensor):
                lt = torch.max(boxes_a_bev[:, None, :2], boxes_b_bev[None, :, :2])  # (n, m, 2)
                rb = torch.min(boxes_a_bev[:, None, 2:4], boxes_b_bev[None, :, 2:4])  # (n, m, 2)
                wh = (rb - lt + offset).clamp(min=0)  # (n, m, 2)
            else:
                lt = np.maximum(boxes_a_bev[:, None, :2], boxes_b_bev[None, :, :2])  # (n, m, 2)
                rb = np.minimum(boxes_a_bev[:, None, 2:4], boxes_b_bev[None, :, 2:4])  # (n, m, 2)
                wh = np.maximum(rb - lt + offset, 0)  # (n, m, 2)
            overlap_bev = wh[:, :, 0] * wh[:, :, 1]  # (n, m)
        elif method == 'depth_bev':
            boxes_a_bev = boxes3d_to_bev_boxes2d(boxes3d_a, bev_axis)
            boxes_b_bev = boxes3d_to_bev_boxes2d(boxes3d_b, bev_axis)
            overlap_bev = _boxes_overlap_bev(boxes_a_bev, boxes_b_bev)
        else:
            assert False
        # compute ios
        overlap = overlap_bev * overlap_d  # (n, m)
        if mode is None:
            return overlap
        if mode in ('iou', 'ioa'):
            volume_a = get_volume(boxes3d_a, method=1, offset=offset)
        if mode in ('iou', 'iob'):
            volume_b = get_volume(boxes3d_b, method=1, offset=offset)
        if mode == 'iou':  # intersection over union
            union = volume_a[:, None] + volume_b[None, :] - overlap  # (n, m)
        elif mode == 'ioa':  # intersection over a
            union = volume_a[:, None]  # (n, m)
        elif mode == 'iob':  # intersection over b
            union = volume_b[None, :]  # (n, m)
        else:
            assert False
        if isinstance(boxes3d_a, torch.Tensor):
            union = torch.max(union, union.new_tensor([eps]))
        else:
            union = np.maximum(union, eps)
        ios = overlap / union  # (n, m)
        return ios

    elif method == 'nearest_bev':
        boxes_a = boxes3d_to_nearest_bev_boxes2d(boxes3d_a, bev_axis)
        boxes_b = boxes3d_to_nearest_bev_boxes2d(boxes3d_b, bev_axis)
        ios = boxes_utils.overlaps(boxes_a, boxes_b, mode=mode, offset=offset, eps=eps)
        return ios

    elif method == 'bev':
        boxes_a_bev = boxes3d_to_bev_boxes2d(boxes3d_a, bev_axis)
        boxes_b_bev = boxes3d_to_bev_boxes2d(boxes3d_b, bev_axis)
        overlap = _boxes_overlap_bev(boxes_a_bev, boxes_b_bev)
        if mode is None:
            return overlap
        if mode in ('iou', 'ioa'):
            size_a = boxes_a_bev[:, 2] * boxes_a_bev[:, 3]
        if mode in ('iou', 'iob'):
            size_b = boxes_b_bev[:, 2] * boxes_b_bev[:, 3]
        if mode == 'iou':  # intersection over union
            union = size_a[:, None] + size_b[None, :] - overlap  # (n, m)
        elif mode == 'ioa':  # intersection over a
            union = size_a[:, None]  # (n, m)
        elif mode == 'iob':  # intersection over b
            union = size_b[None, :]  # (n, m)
        else:
            assert False
        if isinstance(boxes3d_a, torch.Tensor):
            union = torch.max(union, union.new_tensor([eps]))
        else:
            union = np.maximum(union, eps)
        ios = overlap / union  # (n, m)
        return ios

    else:
        assert False


def _boxes_overlap_bev(boxes_a, boxes_b):
    from mmcv.ops import box_iou_rotated

    is_tensor = isinstance(boxes_a, torch.Tensor)
    if not is_tensor:
        boxes_a = torch.from_numpy(boxes_a)
        boxes_b = torch.from_numpy(boxes_b)
    iou = box_iou_rotated(boxes_a, boxes_b)
    size_a = boxes_a[:, 2] * boxes_a[:, 3]
    size_b = boxes_b[:, 2] * boxes_b[:, 3]
    overlap = iou * (size_a[:, None] + size_b[None, :]) / (1 + iou)
    if not is_tensor:
        overlap = overlap.numpy()
    return overlap
