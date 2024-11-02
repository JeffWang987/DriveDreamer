import paths  # noqa F401
from gd_scripts.converters import ImageConverter


def main():
    image_dir = './images/'
    save_path = './giga_data/'
    ImageConverter(image_dir, save_path=save_path, save_version='v0.0.1')()


if __name__ == '__main__':
    main()
