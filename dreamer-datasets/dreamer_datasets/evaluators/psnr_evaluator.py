import logging

import numpy as np
from accelerate import PartialState
from skimage.metrics import peak_signal_noise_ratio


class PSNREvaluator:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, results):
        if PartialState().is_main_process:
            psnr_list = []
            for i, result in enumerate(results):
                data_index = result.get('data_index', i)
                image_true = self.dataset[data_index]['image']
                image_test = result['image']
                if image_test.size != image_true.size:
                    image_true = image_true.resize(image_test.size)
                psnr = calculate_psnr(image_true, image_test)
                psnr_list.append(psnr)
            psnr = sum(psnr_list) / len(psnr_list)
            logging.info('PSNR: {}'.format(psnr))
            return dict(psnr=psnr)


def calculate_psnr(image_true, image_test):
    image_true = np.array(image_true)
    image_test = np.array(image_test)
    psnr = peak_signal_noise_ratio(image_true, image_test, data_range=255)
    return psnr
