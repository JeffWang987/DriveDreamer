import os


def get_root_dir():
    return os.path.abspath(__file__).split('giga_datasets')[0][:-1]


def get_data_dir():
    return os.environ.get('GIGA_DATASETS_DIR', './data/')
