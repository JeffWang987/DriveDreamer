import logging

import lpips as _lpips
import numpy as np
import torch
from accelerate import PartialState


class LPIPSEvaluator:
    def __init__(self, dataset, net='vgg', device='cuda'):
        self.dataset = dataset
        self.net = net
        self.device = device
        self.lpips_loss = None

    def __call__(self, results):
        if PartialState().is_main_process:
            if self.lpips_loss is None:
                self.lpips_loss = _lpips.LPIPS(net=self.net)
                self.lpips_loss.to(self.device)
            lpips_list = []
            for i, result in enumerate(results):
                data_index = result.get('data_index', i)
                image_true = self.dataset[data_index]['image']
                image_test = result['image']
                if image_test.size != image_true.size:
                    image_true = image_true.resize(image_test.size)
                lpips = calculate_lpips(image_true, image_test, self.lpips_loss, self.device)
                lpips_list.append(lpips)
            lpips = sum(lpips_list) / len(lpips_list)
            logging.info('LPIPS: {}'.format(lpips))
            return dict(lpips=lpips)


def calculate_lpips(image_true, image_test, lpips_net, device):
    image_true = np.array(image_true)
    image_test = np.array(image_test)
    image_true = _lpips.im2tensor(image_true).to(device)
    image_test = _lpips.im2tensor(image_test).to(device)
    with torch.no_grad():
        lpips = lpips_net(image_true, image_test)
        lpips = lpips.reshape(-1).item()
    return lpips
