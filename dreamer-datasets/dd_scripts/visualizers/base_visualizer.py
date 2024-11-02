import os

from tqdm import tqdm

from dreamer_datasets import image_utils, load_dataset, utils


class BaseVisualizer:
    def __init__(self, data_path, save_path):
        utils.empty_dir(save_path)
        self.dataset = load_dataset(data_path)
        self.save_path = save_path

    def draw(self, start=0, end=None, step=1, max_num=None):
        end = end or len(self.dataset)
        count = 0
        for i in tqdm(range(start, end, step)):
            data_dict = self.dataset[i]
            image = self.draw_one(data_dict)
            if image is not None:
                if isinstance(image, (list, tuple)):
                    for j in range(len(image)):
                        save_path = os.path.join(self.save_path, 'draw_{:06d}_{}.jpg'.format(i, j))
                        image_utils.save_image(image[j], save_path)
                else:
                    save_path = os.path.join(self.save_path, 'draw_{:06d}.jpg'.format(i))
                    image_utils.save_image(image, save_path)
                count += 1
                if max_num is not None and count >= max_num:
                    return

    def draw_one(self, data_dict):
        raise NotImplementedError
