import os

from tqdm import tqdm

from dreamer_datasets import Dataset, LmdbWriter, PklWriter, load_dataset, utils


class ImageConverter:
    def __init__(self, image_dir, save_path, save_version='v0.0.1'):
        os.makedirs(os.path.join(save_path, save_version), exist_ok=True)
        self.image_dir = image_dir
        self.save_path = save_path
        self.save_version = save_version
        self.save_label_path = os.path.join(save_path, save_version, 'labels')
        self.save_image_path = os.path.join(save_path, save_version, 'images')

    def __call__(self):
        image_paths = utils.list_dir(self.image_dir, recursive=True, exts=['.png', '.jpg', '.jpeg'])
        label_writer = PklWriter(self.save_label_path)
        image_writer = LmdbWriter(self.save_image_path)
        for idx in tqdm(range(len(image_paths))):
            label_writer.write_dict({'data_index': idx})
            image_writer.write_image(idx, image_paths[idx])
        label_writer.write_config()
        image_writer.write_config()
        label_writer.close()
        image_writer.close()
        label_dataset = load_dataset(self.save_label_path)
        image_dataset = load_dataset(self.save_image_path)
        dataset = Dataset([label_dataset, image_dataset])
        dataset.save(os.path.join(self.save_path, self.save_version))


def main():
    image_dir = './images/'
    save_path = './dreamer_data/'
    ImageConverter(image_dir, save_path=save_path, save_version='v0.0.1')()


if __name__ == '__main__':
    main()
