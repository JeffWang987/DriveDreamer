import numpy as np

from dreamer_datasets import ImageVisualizer
from .base_visualizer import BaseVisualizer


class DetVisualizer(BaseVisualizer):
    def __init__(self, data_path, save_path):
        super(DetVisualizer, self).__init__(data_path, save_path)
        self.show_size = 540
        self.categories = []

    def draw_one(self, data_dict):
        assert len(data_dict['boxes']) == len(data_dict['labels'])
        boxes = []
        classes = []
        for i, name in enumerate(data_dict['labels']):
            if name not in self.categories:
                self.categories.append(name)
            boxes.append(data_dict['boxes'][i])
            classes.append(self.categories.index(name) + 1)
        if len(boxes) > 0:
            boxes = np.stack(boxes)
            classes = np.array(classes, dtype=np.int32)
        image = data_dict['image']
        image = ImageVisualizer(image)
        image.draw_boxes(boxes, classes)
        image = image.resize(self.show_size, mode='height')
        return image.get_image()


def main():
    data_path = './data/'
    save_path = './vis_images/'
    visualizer = DetVisualizer(data_path, save_path)
    visualizer.draw()


if __name__ == '__main__':
    main()
