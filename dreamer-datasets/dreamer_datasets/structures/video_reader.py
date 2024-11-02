import math
import queue
import threading
import torch
import cv2

from decord import VideoReader


class VideoReaderCV2:
    def __init__(self, video_path, video_length=None, dst_size=None, queue_size=4):
        self.video_path = video_path
        self.video_length = video_length
        self.dst_size = dst_size
        self.queue_size = queue_size
        self.video = None
        self.frame_queue = None
        self.thread = None
        self.stop = None
        self.cur_frame_idx = -1
        self.open()

    def open(self):
        if self.video is None:
            self.video = cv2.VideoCapture(self.video_path)
            if not self.video.isOpened():
                raise Exception('Ensure file is valid video and system dependencies are up to date.\n')
            self.frame_queue = queue.Queue(self.queue_size)
            self.stop = threading.Event()
            self.thread = threading.Thread(
                target=self.decode_thread, args=(self.video, self.frame_queue, self.stop), daemon=True
            )
            self.thread.start()

    def close(self):
        if self.video is not None:
            self.stop.set()
            self.thread.join()
            self.stop.clear()
            self.video.release()
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
            self.video = None
            self.frame_queue = None
            self.thread = None
            self.stop = None
            self.cur_frame_idx = -1

    def reset(self):
        self.close()
        self.open()

    @property
    def frame_size(self):
        return (
            math.trunc(self.video.get(cv2.CAP_PROP_FRAME_WIDTH)),
            math.trunc(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def fps(self):
        return self.video.get(cv2.CAP_PROP_FPS)

    def read(self):
        frame_idx, frame = self.frame_queue.get()
        if frame_idx is None and frame is None:
            return None
        else:
            self.cur_frame_idx = frame_idx
            return frame

    def decode_thread(self, video, frame_queue, stop):
        frame_idx = 0
        try:
            while not stop.is_set():
                ret, frame = video.read()
                if not ret:
                    break
                if self.dst_size is not None:
                    frame = cv2.resize(frame, self.dst_size, interpolation=cv2.INTER_LINEAR)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_queue.put((frame_idx, frame))
                frame_idx += 1
                if self.video_length is not None and frame_idx >= self.video_length:
                    break
        except Exception:
            stop.set()
        finally:
            frame_queue.put((None, None))


class VideoReaderDecord:
    def __init__(self, video_path, video_length=None, dst_size=None, queue_size=4):
        self.video_path = video_path
        self.video_length = video_length
        self.dst_size = dst_size
        self.queue_size = queue_size
        self.video = None
        self.frame_queue = None
        self.thread = None
        self.stop = None
        self.cur_frame_idx = -1
        self.open()

    def open(self):
        if self.video is None:
            self.video = VideoReader(self.video_path)
            if self.video_length is None:
                self.video_length = len(self.video)
            assert self.video_length <= len(self.video)
            self.frame_queue = queue.Queue(self.queue_size)
            self.stop = threading.Event()
            self.thread = threading.Thread(
                target=self.decode_thread, args=(self.video, self.frame_queue, self.stop), daemon=True
            )
            self.thread.start()

    def close(self):
        if self.video is not None:
            self.stop.set()
            self.thread.join()
            self.stop.clear()
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
            self.video = None
            self.frame_queue = None
            self.thread = None
            self.stop = None
            self.cur_frame_idx = -1

    def reset(self):
        self.close()
        self.open()

    @property
    def fps(self):
        return self.video.get_avg_fps()

    def read(self):
        frame_idx, frame = self.frame_queue.get()
        if frame_idx is None and frame is None:
            return None
        else:
            self.cur_frame_idx = frame_idx
            return frame

    def decode_thread(self, video, frame_queue, stop):
        frame_idx = 0
        try:
            while not stop.is_set():
                frame = video.next()
                frame = frame.numpy() if isinstance(frame, torch.Tensor) else frame.asnumpy()
                if self.dst_size is not None:
                    frame = cv2.resize(frame, self.dst_size, interpolation=cv2.INTER_LINEAR)
                frame_queue.put((frame_idx, frame))
                frame_idx += 1
                if frame_idx >= self.video_length:
                    break
        except Exception:
            stop.set()
        finally:
            frame_queue.put((None, None))
