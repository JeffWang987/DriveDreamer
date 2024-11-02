import numpy as np

from dreamer_datasets import ImageVisualizer
from .base_visualizer import BaseVisualizer


class OnceVisualizer(BaseVisualizer):
    def __init__(self, data_path, save_path):
        super(OnceVisualizer, self).__init__(data_path, save_path)
        self.show_size = 540
        self.show_hdmaps = True
        self.show_boxes = 'boxes3d'
        self.categories = []

    def draw_one(self, data_dict):
        boxes3d_lidar = []
        classes = []
        for i, name in enumerate(data_dict['labels3d']):
            if name not in self.categories:
                self.categories.append(name)
            boxes3d_lidar.append(data_dict['boxes3d'][i])
            classes.append(self.categories.index(name) + 1)
        if len(boxes3d_lidar) > 0:
            boxes3d_lidar = np.stack(boxes3d_lidar)
            classes = np.array(classes, dtype=np.int32)
        calib = data_dict['calib']
        if 'lidar2cam' in calib:
            lidar2cam = calib['lidar2cam']
        elif 'cam2lidar' in calib:
            lidar2cam = np.linalg.inv(calib['cam2lidar'])
        else:
            assert False
        cam_intrinsic = calib['cam_intrinsic']
        image = data_dict['image']
        image = ImageVisualizer(image)
        if self.show_hdmaps and 'image_hdmap' in data_dict:
            image.draw_seg(data_dict['image_hdmap'].resize(image.size), scale=1.0)
        if self.show_boxes == 'boxes3d':
            image.draw_boxes3d_lidar(boxes3d_lidar, classes, lidar2cam, cam_intrinsic)
        elif self.show_boxes == 'boxes2d':
            image.draw_boxes2d_lidar(boxes3d_lidar, classes, lidar2cam, cam_intrinsic)
        image = image.resize(self.show_size, mode='height')
        return image.get_image()


def main():
    data_path = './data/'
    save_path = './vis_images/'
    visualizer = OnceVisualizer(data_path, save_path)
    visualizer.draw()


if __name__ == '__main__':
    main()
