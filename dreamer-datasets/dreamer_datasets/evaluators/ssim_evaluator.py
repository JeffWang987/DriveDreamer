import logging

import numpy as np
from accelerate import PartialState
from skimage.metrics import structural_similarity


class SSIMEvaluator:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, results):
        if PartialState().is_main_process:
            ssim_list = []
            for i, result in enumerate(results):
                data_index = result.get('data_index', i)
                image_true = self.dataset[data_index]['image']
                image_test = result['image']
                if image_test.size != image_true.size:
                    image_true = image_true.resize(image_test.size)
                ssim = calculate_ssim(image_true, image_test)
                ssim_list.append(ssim)
            ssim = sum(ssim_list) / len(ssim_list)
            logging.info('SSIM: {}'.format(ssim))
            return dict(ssim=ssim)


def calculate_ssim(image_true, image_test):
    image_true = np.array(image_true)
    image_test = np.array(image_test)
    if image_true.ndim == 2:
        ssim = structural_similarity(image_true, image_test, data_range=255)
    elif image_true.ndim == 3:
        ssims = []
        for i in range(image_true.shape[2]):
            ssims.append(structural_similarity(image_true[:, :, i], image_test[:, :, i], data_range=255))
        ssim = sum(ssims) / len(ssims)
    else:
        assert False
    return ssim
