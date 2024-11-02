import logging

import numpy as np
import torch
import torchvision.transforms as TF
from accelerate import PartialState
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm


class FIDEvaluator:
    def __init__(self, dataset, batch_size=50, device='cuda', dims=2048, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.dims = dims
        self.num_workers = num_workers

    def __call__(self, results):
        if PartialState().is_main_process:
            annos = []
            for i, result in enumerate(results):
                data_index = result.get('data_index', i)
                image_true = self.dataset[data_index]['image']
                image_test = result['image']
                if image_test.size != image_true.size:
                    image_true = image_true.resize(image_test.size)
                annos.append({'image': image_true})
            fid_value = calculate_fid(
                [annos, results],
                batch_size=self.batch_size,
                device=self.device,
                dims=self.dims,
                num_workers=self.num_workers,
            )
            logging.info('FID: {}'.format(fid_value))
            return dict(fid=fid_value)


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transforms=None):
        self.data_list = data_list
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        img = self.data_list[i]['image']
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(data_list, model, batch_size, device, dims, num_workers):
    """Calculates the activations of the pool_3 layer for all images."""
    model.eval()

    if batch_size > len(data_list):
        print(('Warning: batch size is bigger than the data size. ' 'Setting batch size to data size'))
        batch_size = len(data_list)

    dataset = ImagePathDataset(data_list, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers
    )

    pred_arr = np.empty((len(data_list), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(data_list, model, batch_size, device, dims, num_workers):
    """Calculation of the statistics used by the FID."""
    act = get_activations(data_list, model, batch_size, device, dims, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid(data_list, batch_size, device, dims, num_workers):
    """Calculates the FID."""
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = calculate_activation_statistics(data_list[0], model, batch_size, device, dims, num_workers)
    m2, s2 = calculate_activation_statistics(data_list[1], model, batch_size, device, dims, num_workers)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value
