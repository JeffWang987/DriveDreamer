import io
import os
import pickle
import re
import shutil
from io import BytesIO

import lmdb
import numpy as np
from decord import VideoReader
from PIL import Image, ImageFile, PngImagePlugin

from .. import utils
from .base_dataset import BaseDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_npy_from_stream(stream_):
    """Experimental, may not work!

    :param stream_: io.BytesIO() object obtained by e.g. calling BlockBlobService().get_blob_to_stream() containing the
        binary stream of a standard format .npy file.
    :return: numpy.ndarray
    """
    stream_.seek(0)
    prefix_ = stream_.read(128)  # first 128 bytes seem to be the metadata
    dict_string = re.search('\{(.*?)\}', prefix_[1:].decode())[0]  # noqa W605
    metadata_dict = eval(dict_string)

    array = np.frombuffer(stream_.read(), dtype=metadata_dict['descr']).reshape(metadata_dict['shape'])

    return array


class LmdbDataset(BaseDataset):
    def __init__(self, data_size, data_type, data_name=None, **kwargs):
        super(LmdbDataset, self).__init__(**kwargs)
        self.data_size = data_size
        self.data_type = data_type
        self.data_name = data_name
        self.db = None
        self.reader = None

    @classmethod
    def load(cls, data_or_config):
        from .dataset import load_config

        config = load_config(data_or_config)
        config_path = config.get('config_path', None)
        data_path = config.get('data_path', None)
        data_size = config['data_size']
        data_type = config['data_type']
        data_name = config['data_name']
        return cls(
            config_path=config_path, data_path=data_path, data_size=data_size, data_type=data_type, data_name=data_name
        )

    def save(self, save_path, copy_data=False, store_rel_path=True):
        from .dataset import get_rel_path

        if save_path.endswith('.json'):
            save_config_path = save_path
            save_path = os.path.dirname(save_config_path)
        else:
            save_config_path = os.path.join(save_path, 'config.json')
        config = utils.load_file(self.config_path)
        config['data_type'] = self.data_type
        if self.data_size is not None:
            assert config['data_size'] == self.data_size
        if self.data_name is not None:
            config['_key_names'] = [self.data_name]
            config['data_name'] = self.data_name
        if copy_data:
            os.makedirs(save_path, exist_ok=True)
            os.system('cp -r {}/*.mdb {}'.format(self.data_path, save_path))
        elif store_rel_path:
            config['data_path'] = get_rel_path(self.data_path)
        else:
            config['data_path'] = self.data_path
        utils.save_file(save_config_path, config)

    def open(self):
        if self.reader is None:
            self.db = lmdb.open(self.data_path, readonly=True, lock=False, readahead=False)
            self.reader = self.db.begin()
            if self.data_size is not None:
                assert self.data_size == self.reader.stat()['entries']
            else:
                self.data_size = self.reader.stat()['entries']

    def close(self):
        if self.reader is not None:
            self.db.close()
            self.db = None
            self.reader = None
        super(LmdbDataset, self).close()

    def __len__(self):
        if self.data_size is None:
            self.open()
        return self.data_size

    def _get_data(self, index):
        data = self.reader.get(str(index).encode())
        if self.data_type == 'image':
            data = Image.open(BytesIO(data))
        elif self.data_type == 'video':
            data = VideoReader(BytesIO(data))
        elif self.data_type == 'numpy':
            data = load_npy_from_stream(BytesIO(data))
        elif self.data_type == 'dict':
            data = pickle.loads(data)
        else:
            assert self.data_type == 'raw'
        if self.data_name is not None:
            data_dict = {self.data_name: data}
        else:
            data_dict = data
        return data_dict


class LmdbWriter:
    def __init__(self, data_path, map_size=1, map_unit='TB', commit_interval=1000):
        factors = {
            'TB': 1024 * 1024 * 1024 * 1024,
            'GB': 1024 * 1024 * 1024,
            'MB': 1024 * 1024,
            'KB': 1024,
            'B': 1,
        }
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
        self.data_path = data_path
        self.data_type = None
        self.key_names = []
        self.db = None
        self.writer = None
        self.map_size = map_size * factors[map_unit]
        self.commit_interval = commit_interval
        self._count = 0

    def open(self):
        if self.writer is None:
            self.db = lmdb.open(self.data_path, map_size=self.map_size)
            self.writer = self.db.begin(write=True)

    def close(self):
        if self.writer is not None:
            self.writer.commit()
            self.db.close()
            self.writer = None
            self.db = None

    def _write(self, index, data):
        self.writer.put(str(index).encode(), data)
        self._count += 1
        if self.commit_interval > 0 and self._count % self.commit_interval == 0:
            self.writer.commit()
            self.writer = self.db.begin(write=True)

    def write_image(self, index, image):
        self.open()
        if self.data_type is None:
            self.data_type = 'image'
        else:
            assert self.data_type == 'image'
        if isinstance(image, str):
            data = open(image, 'rb').read()
        elif isinstance(image, Image.Image):
            metadata = PngImagePlugin.PngInfo()
            for key, value in image.info.items():
                if isinstance(key, str) and isinstance(value, str):
                    metadata.add_text(key, value)
            with BytesIO() as output_bytes:
                image.save(output_bytes, format='png', pnginfo=metadata)
                data = output_bytes.getvalue()
        else:
            assert False
        self._write(index, data)

    def write_video(self, index, video):
        self.open()
        if self.data_type is None:
            self.data_type = 'video'
        else:
            assert self.data_type == 'video'
        if isinstance(video, str):
            data = open(video, 'rb').read()
        else:
            assert False
        self._write(index, data)

    def write_numpy(self, index, data):
        self.open()
        if self.data_type is None:
            self.data_type = 'numpy'
        else:
            assert self.data_type == 'numpy'
        assert isinstance(data, np.ndarray)
        imgByteArr = io.BytesIO()
        np.save(imgByteArr, data)
        _ = imgByteArr.seek(0)
        self._write(index, imgByteArr.read())

    def write_dict(self, index, data):
        self.open()
        if self.data_type is None:
            self.data_type = 'dict'
        else:
            assert self.data_type == 'dict'
        assert isinstance(data, dict)
        self.key_names = list(set(self.key_names + list(data.keys())))
        self._write(index, pickle.dumps(data))

    def write_config(self, **kwargs):
        config_path = os.path.join(self.data_path, 'config.json')
        data_name = kwargs.pop('data_name', None)
        if data_name is not None:
            assert self.data_type != 'dict'
        else:
            if self.data_type == 'image':
                data_name = 'image'
            elif self.data_type == 'video':
                data_name = 'video'
            elif self.data_type == 'numpy':
                data_name = 'data'
        if self.data_type == 'dict':
            key_names = self.key_names
        else:
            key_names = [data_name]
        key_names.sort()
        config = {
            '_class_name': 'LmdbDataset',
            '_key_names': key_names,
            'data_size': self.writer.stat()['entries'],
            'data_type': self.data_type,
            'data_name': data_name,
        }
        config.update(kwargs)
        utils.save_file(config_path, config)
