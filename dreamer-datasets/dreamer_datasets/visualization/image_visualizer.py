import os

import cv2
import numpy as np
from PIL import Image, ImageColor

from ..structures import boxes3d_utils, boxes_utils, image_utils, points3d_utils

DEFAULT_COLOR = [
    'blue',
    'green',
    'red',
    'yellow',
    'purple',
    'orange',
    'cyan',
    'gray',
    'brown',
    'deeppink',
]


class ImageVisualizer:
    def __init__(self, image):
        self.image = image_utils.load_image(image, 'np_bgr').copy()

    @property
    def shape(self):
        return self.image.shape

    @property
    def size(self):
        return self.width, self.height

    @property
    def height(self):
        return self.image.shape[0]

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def colormap(self):
        colormap = ImageColor.colormap.copy()
        for key in colormap:
            colormap[key] = ImageColor.getrgb(key)[::-1]
        return colormap

    def get_image(self, dst_format='pil_rgb'):
        if dst_format == 'np_bgr':
            return self.image
        elif dst_format == 'np_rgb':
            return self.image[:, :, ::-1]
        elif dst_format == 'pil_rgb':
            return Image.fromarray(self.image[:, :, ::-1])
        else:
            assert False

    def copy(self):
        return type(self)(self.image)

    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, self.image)

    def resize(self, val, mode, interp=cv2.INTER_LINEAR):
        if mode == 'long':
            scale = val / np.max(self.shape[:2])
        elif mode == 'short':
            scale = val / np.min(self.shape[:2])
        elif mode == 'height':
            scale = val / self.shape[0]
        elif mode == 'width':
            scale = val / self.shape[1]
        elif mode == 'fixed_scale':
            scale = val
        elif mode == 'fixed_size':
            scale = (float(val[0]) / self.shape[0], float(val[1]) / self.shape[1])
        else:
            assert False
        if not isinstance(scale, (tuple, list)):
            scale = (scale, scale)
        self.image = cv2.resize(self.image, None, None, fx=scale[1], fy=scale[0], interpolation=interp)
        return self

    def _get_color(self, i, color=None):
        if color is None:
            color = DEFAULT_COLOR
        if isinstance(color, list):
            color = color[int(i) % len(color)]
        if isinstance(color, str):
            color = self.colormap[color]
        assert isinstance(color, tuple) and len(color) == 3
        return color

    def _get_point_text_org(self, text, org, offset=0):
        x, y = org
        if x > self.image.shape[1] // 2:
            x = x - len(text) * 12 - offset
        else:
            x = x + offset
        if y > self.image.shape[0] // 2:
            y = y - offset
        else:
            y = y + offset
        x = min(self.image.shape[1] - 1, max(0, x))
        y = min(self.image.shape[0] - 1, max(0, y))
        return x, y

    def _get_box_text_org(self, org, offset=0):
        x, y = org
        x = x + offset
        y = y + 15 + offset
        x = min(self.image.shape[1] - 1, max(0, x))
        y = min(self.image.shape[0] - 1, max(0, y))
        return x, y

    def draw_text(self, text, mode='top', char_height=40, char_width=18):
        if mode == 'top':
            cols = self.width // char_width
            rows = (len(text) - 1) // cols + 1
            image_height = (rows + 1) * char_height
            image = np.zeros((image_height, self.width, 3), dtype='uint8')
            for i in range(rows):
                text_i = text[i * cols : (i + 1) * cols]
                cv2.putText(
                    image,
                    text=text_i,
                    org=(0, (i + 1) * char_height),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 255, 0),
                    thickness=2,
                )
            self.image = image_utils.concat_images([image, self.image], direction='vertical')
        else:
            assert False

    def draw_points(self, points, color=None, radius=1, thickness=1, show_num=False):
        # points: (num_points, 2 or 3)
        if len(points) == 0:
            return
        points = np.array(points, dtype=np.int32)
        for i, point in enumerate(points):
            if len(point) == 3:
                x, y, z = point
                color_i = self._get_color(z, color)
            elif len(point) == 2:
                x, y = point
                color_i = self._get_color(i, color)
            else:
                assert False
            cv2.circle(self.image, center=(x, y), color=color_i, radius=radius, thickness=thickness)
            if show_num:
                text = '%d: (%d, %d)' % (i, x, y)
                org = self._get_point_text_org(text, (x, y), offset=radius + thickness)
                cv2.putText(
                    self.image,
                    text=text,
                    org=org,
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1.0,
                    color=color_i,
                    thickness=1,
                )

    def draw_boxes(self, boxes, classes=None, color=None, thickness=2, texts=None, show_num=False):
        # boxes: (num_boxes, 4)
        if len(boxes) == 0:
            return
        boxes = np.array(boxes, dtype=np.int32)
        if show_num:
            assert texts is None
            texts = [str(i) for i in range(len(boxes))]
        for i, box in enumerate(boxes):
            i = classes[i] - 1 if classes is not None else i
            color_i = self._get_color(i, color)
            cv2.rectangle(self.image, (box[0], box[1]), (box[2], box[3]), color=color_i, thickness=thickness)
            if texts is not None:
                text = texts[i % len(texts)]
                org = self._get_box_text_org((box[0], box[1]), offset=thickness)
                cv2.putText(
                    self.image,
                    text=text,
                    org=org,
                    fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    fontScale=1.0,
                    color=color_i,
                    thickness=1,
                )

    def draw_poly_boxes(self, boxes, classes=None, color=None, thickness=2):
        # boxes: (num_boxes, n * 2)
        if len(boxes) == 0:
            return
        boxes = np.array(boxes, dtype=np.int32)
        for i, box in enumerate(boxes):
            i = classes[i] - 1 if classes is not None else i
            color_i = self._get_color(i, color)
            poly_box = box.reshape((-1, 2))
            cv2.polylines(self.image, [poly_box], isClosed=True, color=color_i, thickness=thickness)

    def draw_corners(
        self,
        corners,
        classes=None,
        color=None,
        thickness=2,
        ori_color=None,
        show_ori=False,
        bottom_indexes=None,
        show_num=False,
    ):
        # corners: (num_boxes, 8, 2)
        if len(corners) == 0:
            return
        corners = np.array(corners, dtype=np.int32)
        corners = corners.reshape((corners.shape[0], 8, 2))
        ori_color = ori_color or color
        for i, corner in enumerate(corners):
            i = classes[i] - 1 if classes is not None else i
            color_i = self._get_color(i, color)
            for k in range(4):
                m, n = k, (k + 1) % 4
                cv2.line(
                    self.image,
                    (corner[m, 0], corner[m, 1]),
                    (corner[n, 0], corner[n, 1]),
                    color=color_i,
                    thickness=thickness,
                )
                m, n = k + 4, (k + 1) % 4 + 4
                cv2.line(
                    self.image,
                    (corner[m, 0], corner[m, 1]),
                    (corner[n, 0], corner[n, 1]),
                    color=color_i,
                    thickness=thickness,
                )
                m, n = k, k + 4
                cv2.line(
                    self.image,
                    (corner[m, 0], corner[m, 1]),
                    (corner[n, 0], corner[n, 1]),
                    color=color_i,
                    thickness=thickness,
                )
            if show_ori:
                assert bottom_indexes is not None
                ori_color_i = self._get_color(i, ori_color)
                ori_bottom_start = np.mean(corner[bottom_indexes], axis=0, dtype=np.int32)
                ori_bottom_end = np.mean(corner[bottom_indexes[-2:]], axis=0, dtype=np.int32)
                cv2.line(
                    self.image,
                    (ori_bottom_start[0], ori_bottom_start[1]),
                    (ori_bottom_end[0], ori_bottom_end[1]),
                    color=ori_color_i,
                    thickness=thickness,
                )
            if show_num:
                for k in range(8):
                    text = '%d' % k
                    org = self._get_point_text_org(text, (corner[k, 0], corner[k, 1]), offset=thickness)
                    cv2.putText(
                        self.image,
                        text=text,
                        org=org,
                        fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale=1.0,
                        color=color_i,
                        thickness=1,
                    )

    def draw_seg(self, seg, color=None, scale=0.5):
        if isinstance(seg, Image.Image):
            seg = np.array(seg)
            if seg.ndim == 3:
                seg = seg[:, :, ::-1]
        if seg.ndim == 3:
            seg_img = seg
            mask = np.sum(seg_img, axis=-1) > 0
        elif seg.ndim == 2:
            seg_img = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
            min_ind = int(seg.min())
            max_ind = int(seg.max())
            for i in range(min_ind, max_ind + 1):
                if i <= 0:
                    continue
                color_i = self._get_color(i - 1, color)
                seg_img[seg == i, :] = color_i
            mask = seg > 0
        else:
            assert False
        mask = mask[:, :, np.newaxis]
        self.image = self.image - scale * self.image * mask + scale * seg_img * mask
        self.image = np.clip(self.image, 0, 255).astype(np.uint8)

    def draw_masks(self, masks, color=None, scale=0.5, binary_thresh=0.5):
        if len(masks) == 0:
            return
        masks = (masks >= binary_thresh).astype(np.uint8)
        for i, mask in enumerate(masks):
            color_i = self._get_color(i, color)
            color_i = np.array(color_i).reshape((1, 1, 3))
            mask = mask[:, :, np.newaxis]
            self.image = self.image - scale * self.image * mask + scale * color_i * mask
        self.image = np.clip(self.image, 0, 255).astype(np.uint8)

    def draw_landmarks(self, landmarks, color=(0, 255, 255), radius=2, thickness=2):
        # landmarks: (num_boxes, num_lmk, 3)
        if len(landmarks) == 0:
            return
        landmarks = np.array(landmarks)
        for i in range(landmarks.shape[0]):
            color_j = self._get_color(i, color)
            for j in range(landmarks.shape[1]):
                x = int(landmarks[i, j, 0] + 0.5)
                y = int(landmarks[i, j, 1] + 0.5)
                cv2.circle(self.image, (x, y), color=color_j, radius=radius, thickness=thickness)

    def draw_keypoints(
        self,
        keypoints,
        color=(0, 255, 255),
        radius=2,
        thickness=2,
        skeleton=None,
        skeleton_color=None,
        show_num=False,
    ):
        # keypoints: (num_boxes, num_kps, 3)
        if len(keypoints) == 0:
            return
        keypoints = np.array(keypoints)
        for i in range(keypoints.shape[0]):
            for j in range(keypoints.shape[1]):
                x = int(keypoints[i, j, 0] + 0.5)
                y = int(keypoints[i, j, 1] + 0.5)
                v = keypoints[i, j, 2]
                if v > 0:
                    color_j = self._get_color(j, color)
                    cv2.circle(self.image, (x, y), color=color_j, radius=radius, thickness=thickness)
                    if show_num:
                        text = '%d' % j
                        org = self._get_point_text_org(text, (x, y), offset=radius + thickness)
                        cv2.putText(
                            self.image,
                            text=text,
                            org=org,
                            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale=1.0,
                            color=color_j,
                            thickness=1,
                        )
            if skeleton is not None:
                for j in range(skeleton.shape[0]):
                    p1 = skeleton[j, 0]
                    p2 = skeleton[j, 1]
                    x1 = int(keypoints[i, p1, 0] + 0.5)
                    y1 = int(keypoints[i, p1, 1] + 0.5)
                    v1 = keypoints[i, p1, 2]
                    x2 = int(keypoints[i, p2, 0] + 0.5)
                    y2 = int(keypoints[i, p2, 1] + 0.5)
                    v2 = keypoints[i, p2, 2]
                    if v1 > 0 and v2 > 0:
                        color_j = self._get_color(j, skeleton_color)
                        cv2.line(self.image, (x1, y1), (x2, y2), color=color_j, thickness=thickness)

    def draw_lanes(self, lanes, lane_labels=None, thickness=2):
        for i, lane in enumerate(lanes):
            lane = np.array(lane).astype(np.int32).reshape((-1, 2))
            if lane_labels is not None:
                if 'lane_line' in lane_labels[i]:
                    color = 'blue'
                elif 'road_edge' in lane_labels[i]:
                    color = 'red'
                elif 'stop_line' in lane_labels[i]:
                    color = 'cyan'
                elif 'crosswalk' in lane_labels[i]:
                    color = 'green'
                elif 'zigzag' in lane_labels[i]:
                    color = 'brown'
                else:
                    assert False
            else:
                color = 'blue'
            color = self.colormap[color]
            cv2.polylines(self.image, [lane], isClosed=False, color=color, thickness=thickness)

    def draw_points3d(self, points3d, lidar2cam, cam_intrinsic, **kwargs):
        if len(points3d) == 0:
            return
        p_mat = np.dot(cam_intrinsic, lidar2cam)
        points3d_homo = np.concatenate([points3d[:, :3], np.ones([points3d.shape[0], 1])], axis=1)
        depths = np.dot(points3d_homo, lidar2cam.T)[:, 2]
        points2d = points3d_utils.points3d_to_points2d(points3d[:, :3], p_mat)
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, points2d[:, 0] > 1)
        mask = np.logical_and(mask, points2d[:, 0] < (self.image.shape[1] - 1))
        mask = np.logical_and(mask, points2d[:, 1] > 1)
        mask = np.logical_and(mask, points2d[:, 1] < (self.image.shape[0] - 1))
        points2d = points2d[mask]
        self.draw_points(points2d, **kwargs)

    def draw_boxes3d_camera(self, boxes3d_camera, classes, cam_intrinsic, **kwargs):
        if len(boxes3d_camera) == 0:
            return
        corners3d_camera = boxes3d_utils.boxes3d_to_corners3d(boxes3d_camera, rot_axis=1)
        keep = corners3d_camera[..., 2].mean(-1) > 0
        corners3d_camera = boxes3d_utils.crop_corners3d(corners3d_camera)
        corners2d = boxes3d_utils.corners3d_to_corners2d(corners3d_camera, cam_intrinsic=cam_intrinsic)
        self.draw_corners(corners2d[keep], classes[keep], show_ori=True, bottom_indexes=[2, 3, 6, 7], **kwargs)

    def draw_boxes3d_lidar(self, boxes3d_lidar, classes, lidar2cam, cam_intrinsic, method=2, **kwargs):
        if len(boxes3d_lidar) == 0:
            return
        if method == 1:
            boxes3d_camera = boxes3d_utils.convert_boxes3d(boxes3d_lidar, 'lidar', 'camera', lidar2cam)
            corners3d_camera = boxes3d_utils.boxes3d_to_corners3d(boxes3d_camera, rot_axis=1)
            keep = corners3d_camera[..., 2].mean(-1) > 0
            corners3d_camera = boxes3d_utils.crop_corners3d(corners3d_camera)
            corners2d = boxes3d_utils.corners3d_to_corners2d(corners3d_camera, cam_intrinsic=cam_intrinsic)
        elif method == 2:
            corners3d_lidar = boxes3d_utils.boxes3d_to_corners3d(boxes3d_lidar, rot_axis=2)
            corners3d_one = np.ones(
                [corners3d_lidar.shape[0], corners3d_lidar.shape[1], 1], dtype=corners3d_lidar.dtype
            )
            corners3d_lidar_homo = np.concatenate([corners3d_lidar, corners3d_one], axis=2)
            corners3d_camera = np.dot(corners3d_lidar_homo, lidar2cam.T)[:, :, :3]
            keep = corners3d_camera[..., 2].mean(-1) > 0
            corners3d_camera = boxes3d_utils.crop_corners3d(corners3d_camera)
            corners2d = boxes3d_utils.corners3d_to_corners2d(corners3d_camera, cam_intrinsic=cam_intrinsic)
        else:
            assert False
        self.draw_corners(corners2d[keep], classes[keep], show_ori=True, bottom_indexes=[0, 3, 4, 7], **kwargs)

    def draw_boxes2d_camera(self, boxes3d_camera, classes, cam_intrinsic, **kwargs):
        if len(boxes3d_camera) == 0:
            return
        corners3d_camera = boxes3d_utils.boxes3d_to_corners3d(boxes3d_camera, rot_axis=1)
        keep = corners3d_camera[..., 2].mean(-1) > 0
        corners3d_camera = boxes3d_utils.crop_corners3d(corners3d_camera)
        corners2d = boxes3d_utils.corners3d_to_corners2d(corners3d_camera, cam_intrinsic=cam_intrinsic)
        boxes2d = boxes_utils.corners_to_boxes(corners2d)
        boxes2d = boxes_utils.clip_boxes(boxes2d, self.image.shape[:2])  # (N, 4)
        self.draw_boxes(boxes2d[keep], classes[keep], **kwargs)

    def draw_boxes2d_lidar(self, boxes3d_lidar, classes, lidar2cam, cam_intrinsic, method=2, **kwargs):
        if len(boxes3d_lidar) == 0:
            return
        if method == 1:
            boxes3d_camera = boxes3d_utils.convert_boxes3d(boxes3d_lidar, 'lidar', 'camera', lidar2cam)
            corners3d_camera = boxes3d_utils.boxes3d_to_corners3d(boxes3d_camera, rot_axis=1)
            keep = corners3d_camera[..., 2].mean(-1) > 0
            corners3d_camera = boxes3d_utils.crop_corners3d(corners3d_camera)
            corners2d = boxes3d_utils.corners3d_to_corners2d(corners3d_camera, cam_intrinsic=cam_intrinsic)
        elif method == 2:
            corners3d_lidar = boxes3d_utils.boxes3d_to_corners3d(boxes3d_lidar, rot_axis=2)
            corners3d_one = np.ones(
                [corners3d_lidar.shape[0], corners3d_lidar.shape[1], 1], dtype=corners3d_lidar.dtype
            )
            corners3d_lidar_homo = np.concatenate([corners3d_lidar, corners3d_one], axis=2)
            corners3d_camera = np.dot(corners3d_lidar_homo, lidar2cam.T)[:, :, :3]
            keep = corners3d_camera[..., 2].mean(-1) > 0
            corners3d_camera = boxes3d_utils.crop_corners3d(corners3d_camera)
            corners2d = boxes3d_utils.corners3d_to_corners2d(corners3d_camera, cam_intrinsic=cam_intrinsic)
        else:
            assert False
        boxes2d = boxes_utils.corners_to_boxes(corners2d)
        boxes2d = boxes_utils.clip_boxes(boxes2d, self.image.shape[:2])
        self.draw_boxes(boxes2d[keep], classes[keep], **kwargs)
