from dreamer_datasets import ImageVisualizer
from .base_visualizer import BaseVisualizer


class LaneVisualizer(BaseVisualizer):
    def __init__(self, data_path, save_path):
        super(LaneVisualizer, self).__init__(data_path, save_path)
        self.show_size = 540
        self.categories = []

    def draw_one(self, data_dict):
        image = data_dict['image']
        image = ImageVisualizer(image)
        if 'lanes' in data_dict:
            lanes = data_dict['lanes']
            lane_labels = data_dict.get('lane_labels', None)
            image.draw_lanes(lanes, lane_labels, thickness=2)
        elif 'image_hdmap' in data_dict:
            image_hdmap = data_dict['image_hdmap']
            image.draw_seg(image_hdmap, scale=1.0)
        image = image.resize(self.show_size, mode='height')
        return image.get_image()


def main():
    data_path = './data/'
    save_path = './vis_images/'
    visualizer = LaneVisualizer(data_path, save_path)
    visualizer.draw()


if __name__ == '__main__':
    main()
