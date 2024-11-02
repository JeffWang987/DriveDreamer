import os

import cv2
import numpy as np
import torch
from PIL import Image


def get_rot_mat(angle, src_shape, dst_shape):
    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]
    if isinstance(angle, torch.Tensor):
        angle = angle.view(-1)
        rot_sin = torch.sin(angle)
        rot_cos = torch.cos(angle)
        ones = torch.ones_like(angle)
        zeros = torch.zeros_like(angle)
        t_mat = torch.tensor([1, 0, -src_w / 2.0, 0, 1, -src_h / 2.0, 0, 0, 1], dtype=angle.dtype).reshape((3, 3))
        r_mat = torch.cat([rot_cos, rot_sin, zeros, -rot_sin, rot_cos, zeros, zeros, zeros, ones], dim=0).reshape(
            (3, 3, -1)
        )
        z_mat = torch.tensor([1, 0, dst_w / 2.0, 0, 1, dst_h / 2.0, 0, 0, 1], dtype=angle.dtype).reshape((3, 3))
        rot_mat = torch.einsum('ij,jka,kh->aih', z_mat, r_mat, t_mat)  # (N, 3, 3)
    else:
        angle = np.array(angle).reshape((-1,))
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        ones = np.ones_like(angle)
        zeros = np.zeros_like(angle)
        t_mat = np.array([1, 0, -src_w / 2.0, 0, 1, -src_h / 2.0, 0, 0, 1], dtype=angle.dtype).reshape((3, 3))
        r_mat = np.concatenate([rot_cos, rot_sin, zeros, -rot_sin, rot_cos, zeros, zeros, zeros, ones], axis=0).reshape(
            (3, 3, -1)
        )
        z_mat = np.array([1, 0, dst_w / 2.0, 0, 1, dst_h / 2.0, 0, 0, 1], dtype=angle.dtype).reshape((3, 3))
        rot_mat = np.einsum('ij,jka,kh->aih', z_mat, r_mat, t_mat)  # (N, 3, 3)
    return rot_mat[:, :2]  # (N, 2, 3)


def load_image(image, dst_mode='pil_rgb'):
    if isinstance(image, str):
        if dst_mode == 'np_bgr':
            image = cv2.imread(image, cv2.IMREAD_COLOR)
        elif dst_mode == 'np_rgb':
            image = cv2.imread(image, cv2.IMREAD_COLOR)[:, :, ::-1]
        elif dst_mode == 'np_gray':
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        elif dst_mode == 'pil_rgb':
            image = Image.open(image).convert('RGB')
        elif dst_mode == 'pil_gray':
            image = Image.open(image).convert('L')
        else:
            assert False
    else:
        image = convert_image(image, dst_mode)
    return image


def save_image(image, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(image, np.ndarray):
        cv2.imwrite(save_path, image)
    elif isinstance(image, Image.Image):
        image.save(save_path)
    else:
        assert False


def convert_image(image, dst_mode, src_mode=None):
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            if dst_mode == 'np_bgr':
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif dst_mode == 'np_rgb':
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif dst_mode == 'pil_rgb':
                image = Image.fromarray(image).convert('RGB')
            elif dst_mode == 'pil_gray':
                image = Image.fromarray(image).convert('L')
        elif image.ndim == 3:
            src_mode = src_mode or 'np_bgr'
            if src_mode == 'np_bgr':
                if dst_mode == 'np_rgb':
                    image = image[:, :, ::-1]
                elif dst_mode == 'np_gray':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                elif dst_mode == 'pil_rgb':
                    image = Image.fromarray(image[:, :, ::-1]).convert('RGB')
                elif dst_mode == 'pil_gray':
                    image = Image.fromarray(image[:, :, ::-1]).convert('L')
            elif src_mode == 'np_rgb':
                if dst_mode == 'np_bgr':
                    image = image[:, :, ::-1]
                elif dst_mode == 'np_gray':
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif dst_mode == 'pil_rgb':
                    image = Image.fromarray(image).convert('RGB')
                elif dst_mode == 'pil_gray':
                    image = Image.fromarray(image).convert('L')
            else:
                assert False
    elif isinstance(image, Image.Image):
        if dst_mode == 'np_bgr':
            image = np.array(image.convert('RGB'))[:, :, ::-1]
        elif dst_mode == 'np_rgb':
            image = np.array(image.convert('RGB'))
        elif dst_mode == 'np_gray':
            image = np.array(image.convert('L'))
        elif dst_mode == 'pil_rgb':
            image = image.convert('RGB')
        elif dst_mode == 'pil_gray':
            image = image.convert('L')
        else:
            assert False
    else:
        assert False
    return image


def concat_images(images, direction='horizontal', pad=0, pad_value=0):
    if len(images) == 1:
        return images[0]
    is_pil = isinstance(images[0], Image.Image)
    if is_pil:
        images = [np.array(image) for image in images]
    if direction == 'horizontal':
        height = max([image.shape[0] for image in images])
        width = sum([image.shape[1] for image in images]) + pad * (len(images) - 1)
        new_image = np.full((height, width, images[0].shape[2]), pad_value, dtype=images[0].dtype)
        begin = 0
        for image in images:
            end = begin + image.shape[1]
            new_image[: image.shape[0], begin:end] = image
            begin = end + pad
    elif direction == 'vertical':
        height = sum([image.shape[0] for image in images]) + pad * (len(images) - 1)
        width = max([image.shape[1] for image in images])
        new_image = np.full((height, width, images[0].shape[2]), pad_value, dtype=images[0].dtype)
        begin = 0
        for image in images:
            end = begin + image.shape[0]
            new_image[begin:end, : image.shape[1]] = image
            begin = end + pad
    else:
        assert False
    if is_pil:
        new_image = Image.fromarray(new_image)
    return new_image


def concat_images_grid(images, cols, pad=0, pad_value=0):
    new_images = []
    while len(images) > 0:
        new_image = concat_images(images[:cols], pad=pad, pad_value=pad_value)
        new_images.append(new_image)
        images = images[cols:]
    new_image = concat_images(new_images, direction='vertical', pad=pad, pad_value=pad_value)
    return new_image


def pad_image(image, dst_size, mode='top_left', pad_value=0):
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        image = np.array(image)
    dst_width, dst_height = dst_size
    height = image.shape[0]
    width = image.shape[1]
    assert dst_height >= height and dst_width >= width
    if image.ndim == 3:
        new_image = np.full((dst_height, dst_width, image.shape[2]), fill_value=pad_value, dtype=image.dtype)
    elif image.ndim == 2:
        new_image = np.full((dst_height, dst_width), fill_value=pad_value, dtype=image.dtype)
    else:
        assert False
    if mode == 'top_left':
        y1 = x1 = 0
    elif mode == 'center':
        y1 = (dst_height - height) // 2
        x1 = (dst_width - width) // 2
    else:
        assert False
    new_image[y1 : y1 + height, x1 : x1 + width] = image
    if is_pil:
        new_image = Image.fromarray(new_image)
    return new_image


def get_resample(resample):
    if resample == 'nearest':
        resample = Image.Resampling.NEAREST
    elif resample == 'bilinear':
        resample = Image.Resampling.BILINEAR
    elif resample == 'bicubic':
        resample = Image.Resampling.BICUBIC
    elif resample == 'lanczos':
        resample = Image.Resampling.LANCZOS
    return resample


def get_image_size(src_size, dst_size, mode, max_size=None, multiple=None):
    width, height = src_size
    if mode in ('long', 'short', 'height', 'width'):
        if isinstance(dst_size, (list, tuple)):
            assert dst_size[0] == dst_size[1]
            dst_size = dst_size[0]
        if mode == 'long':
            scale = float(dst_size) / max(height, width)
        elif mode == 'short':
            scale = float(dst_size) / min(height, width)
        elif mode == 'height':
            scale = float(dst_size) / height
        elif mode == 'width':
            scale = float(dst_size) / width
        dst_height = int(round(height * scale))
        dst_width = int(round(width * scale))
    elif mode in ('fixed', 'outer_fit', 'inner_fit'):
        if isinstance(dst_size, (list, tuple)):
            dst_width, dst_height = dst_size
        else:
            dst_width, dst_height = dst_size, dst_size
        if mode == 'outer_fit':
            if float(dst_height) / height > float(dst_width) / width:
                dst_height = int(round(float(dst_width) / width * height))
            else:
                dst_width = int(round(float(dst_height) / height * width))
        elif mode == 'inner_fit':
            if float(dst_height) / height < float(dst_width) / width:
                dst_height = int(round(float(dst_width) / width * height))
            else:
                dst_width = int(round(float(dst_height) / height * width))
    elif mode == 'area':
        if isinstance(dst_size, (list, tuple)):
            dst_width, dst_height = dst_size
        else:
            dst_width, dst_height = dst_size, dst_size
        aspect_ratio = float(height) / float(width)
        dst_area = dst_height * dst_width
        dst_height = int(round(np.sqrt(dst_area * aspect_ratio)))
        dst_width = int(round(np.sqrt(dst_area / aspect_ratio)))
    else:
        assert False
    if max_size is not None and max(dst_height, dst_width) > max_size:
        if dst_height > dst_width:
            dst_width = int(round(float(max_size) / dst_height * dst_width))
            dst_height = max_size
        else:
            dst_height = int(round(float(max_size) / dst_width * dst_height))
            dst_width = max_size
    if multiple is not None:
        dst_height = int(round(dst_height / multiple)) * multiple
        dst_width = int(round(dst_width / multiple)) * multiple
    return dst_width, dst_height


def resize_image(image, dst_size, mode, max_size=None, multiple=None, resample=None):
    is_pil = isinstance(image, Image.Image)
    if not is_pil:
        image = Image.fromarray(image)
    dst_size = get_image_size(image.size, dst_size, mode, max_size, multiple)
    image = image.resize(dst_size, resample=get_resample(resample))
    if not is_pil:
        image = np.array(image)
    return image


def rotate_image(image, rot_mat, dst_shape, interp=cv2.INTER_LINEAR):
    if len(rot_mat.shape) == 3:
        assert rot_mat.shape[0] == 1
        rot_mat = rot_mat[0]
    dst_h, dst_w = dst_shape[:2]
    image = cv2.warpAffine(image, rot_mat, (dst_w, dst_h), flags=interp)
    return image
