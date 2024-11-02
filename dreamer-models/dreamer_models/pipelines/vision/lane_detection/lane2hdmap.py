import cv2
import numpy as np
from PIL import Image


class Lane2Hdmap:
    def __init__(self, thickness):
        self.thickness = thickness

    def __call__(self, lanes, lane_labels, image_size, hdmap_size=None):
        hdmap_size = hdmap_size or image_size
        image_width, image_height = image_size
        hdmap_width, hdmap_height = hdmap_size
        image_hdmap = np.zeros((hdmap_height, hdmap_width, 3), dtype=np.uint8)  # RGB
        for i, lane in enumerate(lanes):
            lane_label = lane_labels[i] if lane_labels is not None else None
            color = get_lane_color(lane_label)
            lane = np.array(lane, dtype=np.float32).reshape((-1, 2)).copy()
            lane[:, 0] /= image_width / hdmap_width
            lane[:, 1] /= image_height / hdmap_height
            cv2.polylines(image_hdmap, [lane.astype(np.int32)], isClosed=False, color=color, thickness=self.thickness)
        image_hdmap = Image.fromarray(image_hdmap)
        return image_hdmap


def get_lane_color(lane_label):
    lane_color = [255, 255, 255]  # RGB
    if lane_label is None or len(lane_label) == 0:
        return lane_color
    if 'lane_line' in lane_label:
        lane_color[0] = 32
        if 'single' in lane_label and 'solid' in lane_label:
            lane_color[1] = 32
        elif 'single' in lane_label and 'dot' in lane_label:
            lane_color[1] = 48
        elif 'double' in lane_label and 'solid' in lane_label:
            lane_color[1] = 64
        elif 'double' in lane_label and 'dot' in lane_label:
            lane_color[1] = 80
        elif 'double' in lane_label and 'solid_dot' in lane_label:
            lane_color[1] = 96
        elif 'diamoid' in lane_label and 'solid_dot' in lane_label:
            lane_color[1] = 112
        elif 'diamoid' in lane_label and 'solid' in lane_label:
            lane_color[1] = 128
        elif 'diamoid' in lane_label and 'dot' in lane_label:
            lane_color[1] = 144
        else:
            assert False, lane_label
    elif 'road_edge' in lane_label:
        lane_color[0] = 64
    elif 'stop_line' in lane_label:
        lane_color[0] = 96
        if 'single' in lane_label and 'solid' in lane_label:
            lane_color[1] = 32
        elif 'single' in lane_label and 'dot' in lane_label:
            lane_color[1] = 48
        elif 'double' in lane_label and 'solid' in lane_label:
            lane_color[1] = 64
        elif 'double' in lane_label and 'dot' in lane_label:
            lane_color[1] = 80
        elif 'double' in lane_label and 'solid_dot' in lane_label:
            lane_color[1] = 96
    elif 'crosswalk' in lane_label:
        lane_color[0] = 128
    elif 'chevron' in lane_label:
        lane_color[0] = 160
    elif 'zigzag' in lane_label:
        lane_color[0] = 192
    else:
        assert False, lane_label
    if 'edge' in lane_label:
        lane_color[0] += 8
    elif 'semantic' in lane_label:
        lane_color[0] += 16
    if 'white' in lane_label:
        lane_color[2] = 32
    elif 'yellow' in lane_label:
        lane_color[2] = 64
    elif 'other' in lane_label:
        lane_color[2] = 224
    return lane_color
