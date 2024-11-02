import numpy as np

from dreamer_datasets import ImageVisualizer, image_utils
from .base_visualizer import BaseVisualizer


class NuScenesLidarVisualizer(BaseVisualizer):
    def __init__(self, data_path, save_path):
        super(NuScenesLidarVisualizer, self).__init__(data_path, save_path)
        self.cam_types = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        self.show_size = 1080
        self.show_boxes = 'boxes3d'
        self.show_points = False
        self.categories = []

    def draw_one(self, data_dict):
        boxes3d_lidar = []
        classes = []
        for i, name in enumerate(data_dict['labels3d']):
            if data_dict['num_lidar_pts'][i] == 0:
                continue
            if name not in self.categories:
                self.categories.append(name)
            boxes3d_lidar.append(data_dict['boxes3d'][i])
            classes.append(self.categories.index(name) + 1)
        if len(boxes3d_lidar) > 0:
            boxes3d_lidar = np.stack(boxes3d_lidar)
            classes = np.array(classes, dtype=np.int32)
        points3d = data_dict['points3d']
        images = []
        for cam_type in self.cam_types:
            cam_name = cam_type.lower()[4:]
            cam_calib = data_dict['calib'][cam_type.lower()]
            lidar2cam = np.linalg.inv(cam_calib['cam2lidar'])
            cam_intrinsic = cam_calib['cam_intrinsic']
            image = data_dict['image_' + cam_name]
            image = ImageVisualizer(image)
            if self.show_boxes == 'boxes3d':
                image.draw_boxes3d_lidar(boxes3d_lidar, classes, lidar2cam, cam_intrinsic)
            elif self.show_boxes == 'boxes2d':
                image.draw_boxes2d_lidar(boxes3d_lidar, classes, lidar2cam, cam_intrinsic)
            if self.show_points:
                image.draw_points3d(points3d, lidar2cam, cam_intrinsic, color='green')
            images.append(image.get_image())
        image_front = image_utils.concat_images([images[1], images[0], images[2]], pad=2)
        image_back = image_utils.concat_images([images[4], images[3], images[5]], pad=2)
        image = image_utils.concat_images([image_front, image_back], 'vertical', pad=2)
        image = ImageVisualizer(image)
        image = image.resize(self.show_size, mode='height')
        return image.get_image()


class NuScenesCamVisualizer(BaseVisualizer):
    def __init__(self, data_path, save_path):
        super(NuScenesCamVisualizer, self).__init__(data_path, save_path)
        self.show_size = 540
        self.show_hdmaps = True
        self.show_boxes = 'boxes3d'
        self.show_centers = False
        self.categories = []

    def draw_one(self, data_dict):
        boxes3d_camera = []
        classes = []
        for i, name in enumerate(data_dict['labels']):
            if name not in self.categories:
                self.categories.append(name)
            boxes3d_camera.append(data_dict['boxes3d'][i])
            classes.append(self.categories.index(name) + 1)
        if len(boxes3d_camera) > 0:
            boxes3d_camera = np.stack(boxes3d_camera)
            classes = np.array(classes, dtype=np.int32)
        cam_intrinsic = data_dict['calib']['cam_intrinsic']
        image = data_dict['image']
        image = ImageVisualizer(image)
        if self.show_hdmaps and 'image_hdmap' in data_dict:
            image.draw_seg(data_dict['image_hdmap'].resize(image.size), scale=1.0)
        if self.show_boxes == 'boxes3d':
            image.draw_boxes3d_camera(boxes3d_camera, classes, cam_intrinsic)
        elif self.show_boxes == 'boxes2d':
            image.draw_boxes2d_camera(boxes3d_camera, classes, cam_intrinsic)
        image = image.resize(self.show_size, mode='height')
        return image.get_image()


def main():
    data_path = './data/'
    save_path = './vis_images/'
    visualizer = NuScenesCamVisualizer(data_path, save_path)
    visualizer.draw()


if __name__ == '__main__':
    main()
