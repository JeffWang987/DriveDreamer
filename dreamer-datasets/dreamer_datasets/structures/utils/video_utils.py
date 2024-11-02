import os

import imageio
import numpy as np
import torch
from decord import VideoReader
from PIL import Image


def load_video(video_path):
    video = VideoReader(video_path)
    indexes = np.arange(len(video), dtype=int)
    frames = video.get_batch(indexes)
    frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
    frames = [Image.fromarray(frame) for frame in frames]
    return frames


def save_video(save_path, frames, fps, **kwargs):
    imageio.mimsave(save_path, frames, fps=fps, **kwargs)


def get_fps(video_path):
    video = VideoReader(video_path)
    return video.get_avg_fps()


def sample_video(video, indexes, method=2):
    if method == 1:
        frames = video.get_batch(indexes)
        frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
    elif method == 2:
        max_idx = indexes.max() + 1
        all_indexes = np.arange(max_idx, dtype=int)
        frames = video.get_batch(all_indexes)
        frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
        frames = frames[indexes]
    else:
        assert False
    return frames


def add_music_to_video(src_video_path, dst_video_path, save_video_path=None):
    aud_path = src_video_path[:-4] + '.m4a'
    if save_video_path is None:
        save_video_path = dst_video_path[:-4] + '_music' + dst_video_path[-4:]
    if os.path.exists(save_video_path):
        cmd = 'rm -f %s' % save_video_path
        os.system(cmd)
    cmd = 'ffmpeg -i %s -vn -y -acodec copy %s' % (src_video_path, aud_path)
    os.system(cmd)
    cmd = 'ffmpeg -i %s -i %s -vcodec copy -acodec copy %s' % (dst_video_path, aud_path, save_video_path)
    os.system(cmd)
    cmd = 'rm -f %s' % aud_path
    os.system(cmd)
