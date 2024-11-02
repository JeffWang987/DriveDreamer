import datetime
import os
import time

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, ProjectConfiguration, set_seed
from dreamer_datasets import DefaultCollator, load_dataset

from .. import utils
from ..configs import load_config
from ..transforms import build_transform


class Tester:
    def __init__(
        self,
        project_dir,
        mixed_precision=None,
        log_interval=100,
        seed=6666,
        **kwargs,
    ):
        assert seed > 0
        set_seed(seed)
        project_config = ProjectConfiguration(
            project_dir=project_dir,
            logging_dir=os.path.join(project_dir, 'logs'),
        )
        dataloader_config = DataLoaderConfiguration(
            split_batches=False,
            even_batches=False,
        )
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            project_config=project_config,
            dataloader_config=dataloader_config,
        )
        os.makedirs(self.logging_dir, exist_ok=True)
        if self.is_main_process:
            log_name = 'test_{}.log'.format(utils.get_cur_time())
            self.logger = utils.create_logger(os.path.join(self.logging_dir, log_name))
        else:
            self.logger = utils.create_logger()

        self.log_interval = log_interval
        self.seed = seed
        self.kwargs = kwargs

        self._dataloaders = []
        self._models = []

        self._cur_step = 0
        self._start_tic = None
        self._step_tic = None

    @property
    def project_dir(self):
        return self.accelerator.project_dir

    @property
    def logging_dir(self):
        return self.accelerator.logging_dir

    @property
    def model_dir(self):
        return os.path.join(self.project_dir, 'models')

    @property
    def distributed_type(self):
        return self.accelerator.distributed_type

    @property
    def num_processes(self):
        return self.accelerator.num_processes

    @property
    def process_index(self):
        return self.accelerator.process_index

    @property
    def local_process_index(self):
        return self.accelerator.local_process_index

    @property
    def is_main_process(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main_process(self):
        return self.accelerator.is_local_main_process

    @property
    def is_last_process(self):
        return self.accelerator.is_last_process

    @property
    def mixed_precision(self):
        return self.accelerator.mixed_precision

    @property
    def device(self):
        return self.accelerator.device

    @property
    def dtype(self):
        return torch.float16 if self.mixed_precision == 'fp16' else torch.float32

    @property
    def dataloaders(self):
        return self._dataloaders

    @property
    def dataloader(self):
        return self._dataloaders[0]

    @property
    def models(self):
        return self._models

    @property
    def model(self):
        return self._models[0]

    @property
    def data_size(self):
        return len(self.dataloader.dataset)

    @property
    def batch_size(self):
        if self.dataloader.batch_sampler is not None:
            batch_sampler = self.dataloader.batch_sampler
        else:
            batch_sampler = self.dataloader.sampler
        while True:
            if hasattr(batch_sampler, 'batch_sampler'):
                batch_sampler = batch_sampler.batch_sampler
            else:
                break
        batch_size = batch_sampler.batch_size
        return batch_size * self.num_processes

    @property
    def epoch_size(self):
        return len(self.dataloader)

    @property
    def cur_step(self):
        return self._cur_step

    def print(self, msg, *args, **kwargs):
        if self.is_main_process:
            self.logger.info(msg, *args, **kwargs)

    @classmethod
    def load(cls, config_or_path):
        config = load_config(config_or_path).copy()
        tester = cls(project_dir=config.project_dir, **config.test)
        tester.prepare(
            dataloaders=config.dataloaders.test,
            models=config.models.test if hasattr(config.models, 'test') else config.models,
        )
        return tester

    def get_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            checkpoints = os.listdir(self.model_dir)
            checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1]))
            if len(checkpoints) > 0:
                checkpoint = os.path.join(self.model_dir, checkpoints[-1])
            else:
                return None
        if not isinstance(checkpoint, list):
            checkpoint = [checkpoint]
        for i in range(len(checkpoint)):
            if checkpoint[i].startswith('checkpoint'):
                checkpoint[i] = os.path.join(self.model_dir, checkpoint[i])
            assert os.path.exists(checkpoint[i])
        return checkpoint if len(checkpoint) > 1 else checkpoint[0]

    def get_dataloaders(self, data_config):
        dataset = load_dataset(data_config.data_or_config)
        filter_cfg = data_config.get('filter', None)
        if filter_cfg is not None:
            dataset.filter(**filter_cfg)
        transform = build_transform(data_config.transform)
        dataset.set_transform(transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=DefaultCollator(),
            batch_size=data_config.batch_size_per_gpu,
            num_workers=data_config.num_workers,
        )
        return dataloader

    def get_models(self, *args, **kwargs):
        raise NotImplementedError

    def prepare(self, dataloaders, models):
        self._dataloaders = utils.as_list(self.get_dataloaders(dataloaders))
        self._models = utils.as_list(self.get_models(models))
        self._dataloaders = utils.as_list(self.accelerator.prepare(*self._dataloaders))

    def test(self):
        raise NotImplementedError

    def print_before_test(self):
        msg = 'num_processes: {}'.format(self.num_processes)
        msg += ', process_index: {}'.format(self.process_index)
        msg += ', data_size: {}'.format(self.data_size)
        msg += ', batch_size: {}'.format(self.batch_size)
        msg += ', epoch_size: {}'.format(self.epoch_size)
        self.logger.info(msg)
        self._step_tic = self._start_tic = time.time()

    def print_step(self):
        if self.cur_step % self.log_interval == 0:
            time_cost = time.time() - self._step_tic
            self._step_tic = time.time()
            speed = self.log_interval * self.batch_size / time_cost
            eta_sec = max(0, time_cost / self.log_interval * (self.epoch_size - self.cur_step))
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            msg = 'Node[%d] Step[%d/%d]' % (self.process_index, self.cur_step, self.epoch_size)
            msg += ' eta: %s, time: %.3f, speed: %.3f' % (eta_str, time_cost, speed)
            self.logger.info(msg)

    def print_after_test(self):
        time_cost = time.time() - self._start_tic
        time_cost = str(datetime.timedelta(seconds=int(time_cost)))
        self.logger.info('Node[%d] Total_time: %s' % (self.process_index, time_cost))
