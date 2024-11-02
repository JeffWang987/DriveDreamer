import datetime
import json
import math
import os
import shutil
import time
import functools

import diffusers
import torch
import transformers
from accelerate import Accelerator, DistributedType, skip_first_batches
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, ProjectConfiguration, set_seed, release_memory
from diffusers.training_utils import EMAModel
from diffusers.utils import WEIGHTS_NAME
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import apply_activation_checkpointing
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch import nn
from dreamer_datasets import DefaultCollator, load_dataset
from dreamer_models import utils as dm_utils
from dreamer_models.nn import ModuleDict

from .. import utils
from ..configs import load_config
from ..optimizers import build_optimizer
from ..samplers import build_sampler
from ..schedulers import build_scheduler
from ..transforms import build_transform


class Trainer:
    def __init__(
        self,
        project_dir,
        max_epochs=0,
        max_steps=0,
        gradient_accumulation_steps=1,
        mixed_precision=None,
        loss_nan_total_limit=100,
        checkpoint_interval=1,
        checkpoint_total_limit=-1,
        checkpoint_keeps=None,
        checkpoint_save_optimizer=False,
        log_with=None,
        log_interval=100,
        with_ema=False,
        activation_checkpointing=False,
        activation_class_names=None,
        find_unused_parameters=False,
        broadcast_buffers=True,
        allow_tf32=True,
        seed=6666,
        **kwargs,
    ):
        assert seed > 0
        set_seed(seed)
        if project_dir.endswith('/'):
            project_dir = project_dir[:-1]
        project_name = os.path.basename(project_dir)
        project_config = ProjectConfiguration(
            project_dir=project_dir,
            logging_dir=os.path.join(project_dir, 'logs'),
        )
        dataloader_config = DataLoaderConfiguration(
            split_batches=False,
        )
        # ref: https://huggingface.co/docs/accelerate/basic_tutorials/overview
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=log_with,
            project_config=project_config,
            dataloader_config=dataloader_config,
            kwargs_handlers=[
                DistributedDataParallelKwargs(
                    find_unused_parameters=find_unused_parameters,
                    broadcast_buffers=broadcast_buffers,
                )
            ],
        )
        self.accelerator.init_trackers(project_name)
        # ref: https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
        # with TF32 enabled, the speed is ~7x faster on A100, and that relative error compared to double precision is approximately 2 orders of magnitude larger.
        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if self.is_main_process:
            os.makedirs(self.logging_dir, exist_ok=True)
            os.makedirs(self.model_dir, exist_ok=True)
            log_name = 'train_{}.log'.format(utils.get_cur_time())
            self.logger = utils.create_logger(os.path.join(self.logging_dir, log_name))
        else:
            self.logger = utils.create_logger()

        self.loss_nan_total_limit = loss_nan_total_limit
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_total_limit = checkpoint_total_limit
        self.checkpoint_keeps = checkpoint_keeps
        self.checkpoint_save_optimizer = checkpoint_save_optimizer
        self.log_interval = log_interval
        self.activation_checkpointing = activation_checkpointing
        self.activation_class_names = activation_class_names
        self.seed = seed
        self.kwargs = kwargs

        self.with_ema = with_ema
        self.ema_models = []

        self._dataloaders = []
        self._models = []
        self._optimizers = []
        self._schedulers = []

        if max_epochs > 0:
            assert max_steps == 0
            by_epoch = True
        else:
            assert max_epochs > 0
            by_epoch = False
        self._by_epoch = by_epoch
        self._max_epochs = max_epochs
        self._max_steps = max_steps
        self._cur_step = 0
        self._skip_batches = 0

        self._start_tic = None
        self._epoch_tic = None
        self._step_tic = None
        self._outputs = dict()
        self._loss_nan_count = 0

        if self.distributed_type == DistributedType.DEEPSPEED:
            # ref: https://huggingface.co/docs/transformers/v4.43.3/deepspeed
            # From DeepSpeed==0.8.3 on, if you want to use offload, you’ll also need to the following to the top level configuration because offload works best with DeepSpeed’s CPU Adam optimizer.
            self.accelerator.state.deepspeed_plugin.deepspeed_config['zero_force_ds_cpu_optimizer'] = False
        self.accelerator.register_for_checkpointing(self)
        self.accelerator.register_save_state_pre_hook(self.save_model_hook)
        self.accelerator.register_load_state_pre_hook(self.load_model_hook)
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

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
        if self.mixed_precision == 'fp16':
            return torch.float16
        if self.mixed_precision == 'bf16':
            return torch.bfloat16
        else:
            return torch.float32

    @property
    def gradient_accumulation_steps(self):
        return self.accelerator.gradient_accumulation_steps

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
    def optimizers(self):
        return self._optimizers

    @property
    def optimizer(self):
        return self._optimizers[0]

    @property
    def schedulers(self):
        return self._schedulers

    @property
    def scheduler(self):
        return self._schedulers[0]

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
        return batch_size * self.num_processes * self.gradient_accumulation_steps

    @property
    def epoch_size(self):
        return int(math.ceil((len(self.dataloader) + self._skip_batches) / self.gradient_accumulation_steps))

    @property
    def max_epochs(self):
        if self._max_epochs > 0:
            return self._max_epochs
        else:
            return int(math.ceil(self._max_steps / self.epoch_size))

    @property
    def max_steps(self):
        if self._max_steps > 0:
            return self._max_steps
        else:
            return self._max_epochs * self.epoch_size

    @property
    def cur_epoch(self):
        return int(math.ceil(self.cur_step / self.epoch_size))

    @property
    def cur_step(self):
        return self._cur_step

    def print(self, msg, *args, **kwargs):
        if self.is_main_process:
            self.logger.info(msg, *args, **kwargs)

    def state_dict(self):
        return {'step': self._cur_step}

    def load_state_dict(self, state_dict):
        self._cur_step = state_dict['step']

    @classmethod
    def load(cls, config_or_path):
        config = load_config(config_or_path).copy()
        trainer = cls(project_dir=config.project_dir, **config.train)
        trainer.prepare(
            dataloaders=config.dataloaders.train,
            models=config.models.train if hasattr(config.models, 'train') else config.models,
            optimizers=config.optimizers,
            schedulers=config.schedulers,
        )
        return trainer

    def save_config(self, config):
        if not self.is_main_process:
            return
        config = load_config(config)
        config_path = os.path.join(self.project_dir, 'config.json')
        config.save(config_path)

    def load_checkpoint(self, checkpoint, models, strict=True):
        if checkpoint is None:
            return
        checkpoint = self.get_checkpoint(checkpoint)
        if not isinstance(checkpoint, list):
            checkpoint = [checkpoint]
        if not isinstance(models, list):
            models = [models]
        for i in range(len(checkpoint)):
            config_path = os.path.join(checkpoint[i], 'config.json')
            config = json.load(open(config_path, 'r'))
            class_name = config['_class_name']
            self.logger.info(f'Load {class_name} from {checkpoint[i]}')
            state_dict = dm_utils.load_state_dict(checkpoint[i])
            flag = False
            for model in models:
                if model.__class__.__name__ == class_name:
                    mes = model.load_state_dict(state_dict, strict=strict)
                    if self.is_main_process and not strict:
                        self.logger.info(mes)
                    flag = True
                    break
            if not flag:
                raise ValueError('No model loaded by {checkpoint[i]}')

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

    def remove_checkpoint(self, total_limit=None):
        if not self.is_main_process:
            return
        total_limit = total_limit or self.checkpoint_total_limit
        checkpoints = os.listdir(self.model_dir)
        checkpoints = [d for d in checkpoints if d.startswith('checkpoint')]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1]))
        if self.checkpoint_keeps is not None:
            new_checkpoints = []
            for checkpoint in checkpoints:
                if self._by_epoch:
                    checkpoint_id = int(checkpoint.split('_')[-3])
                else:
                    checkpoint_id = int(checkpoint.split('_')[-1])
                if checkpoint_id not in self.checkpoint_keeps:
                    new_checkpoints.append(checkpoint)
            checkpoints = new_checkpoints
        if len(checkpoints) >= total_limit > 0:
            num_to_remove = len(checkpoints) - self.checkpoint_total_limit + 1
            for checkpoint in checkpoints[:num_to_remove]:
                checkpoint = os.path.join(self.model_dir, checkpoint)
                self.logger.info('Remove checkpoint {}'.format(checkpoint))
                shutil.rmtree(checkpoint)
            checkpoints = checkpoints[num_to_remove:]
        if not self.checkpoint_save_optimizer:
            for checkpoint in checkpoints:
                if self.distributed_type == DistributedType.DEEPSPEED:
                    checkpoint = os.path.join(self.model_dir, checkpoint, 'pytorch_model')
                    if os.path.exists(checkpoint):
                        shutil.rmtree(checkpoint)
                else:
                    for i in range(len(self.optimizers)):
                        optimizer_name = 'optimizer.bin' if i == 0 else f'optimizer_{i}.bin'
                        checkpoint_i = os.path.join(self.model_dir, checkpoint, optimizer_name)
                        if os.path.exists(checkpoint_i):
                            os.remove(checkpoint_i)

    def resume(self, checkpoint=None):
        checkpoint = self.get_checkpoint(checkpoint)
        if checkpoint is None:
            return
        self.accelerator.load_state(checkpoint)
        if self.dataloader.batch_sampler is not None:
            sampler = self.dataloader.batch_sampler
        else:
            sampler = self.dataloader.sampler
        while True:
            if hasattr(sampler, 'batch_sampler'):
                sampler = sampler.batch_sampler
            elif hasattr(sampler, 'sampler'):
                sampler = sampler.sampler
            else:
                break
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(int(math.floor(self.cur_step / self.epoch_size)))
            skip_batches = (self.cur_step % self.epoch_size) * self.gradient_accumulation_steps
        else:
            skip_batches = self.cur_step * self.gradient_accumulation_steps
        if skip_batches > 0:
            for i in range(len(self._dataloaders)):
                self._dataloaders[i] = skip_first_batches(self._dataloaders[i], skip_batches)
        self._skip_batches = skip_batches

    def get_dataloaders(self, data_config):
        dataset = load_dataset(data_config.data_or_config)
        batch_size = data_config.batch_size_per_gpu * self.num_processes * self.gradient_accumulation_steps
        filter_cfg = data_config.get('filter', None)
        if filter_cfg is not None:
            dataset.filter(**filter_cfg)
        transform = build_transform(data_config.transform)
        dataset.set_transform(transform)
        sampler_cfg = data_config.get('sampler', {'type': 'DefaultSampler'})
        sampler = build_sampler(sampler_cfg, dataset=dataset, batch_size=batch_size)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            collate_fn=DefaultCollator(),
            batch_size=data_config.batch_size_per_gpu,
            num_workers=data_config.num_workers,
        )
        return dataloader

    def get_models(self, *args, **kwargs):
        raise NotImplementedError

    def get_optimizers(self, optimizers):
        optimizers = utils.as_list(optimizers)
        for i in range(len(optimizers)):
            if isinstance(optimizers[i], dict):
                if len(optimizers) == 1 and len(self.models) > 1:
                    params = []
                    for model in self.models:
                        params += list(model.parameters())
                elif len(optimizers) == len(self.models):
                    params = self.models[i].parameters()
                else:
                    assert False
                optimizers[i] = build_optimizer(optimizers[i], params=params)
        return optimizers

    def get_schedulers(self, schedulers):
        schedulers = utils.as_list(schedulers)
        assert len(schedulers) == len(self.optimizers)
        for i in range(len(schedulers)):
            if isinstance(schedulers[i], dict):
                schedulers[i] = build_scheduler(
                    schedulers[i],
                    optimizer=self.optimizers[i],
                    epoch_size=self.epoch_size,
                    max_epochs=self.max_epochs,
                    max_steps=self.max_steps,
                )
        return schedulers

    def set_ema_models(self):
        if self.with_ema:
            self.ema_models = [EMAModel(model.parameters()) for model in self.models]

    def apply_activation_checkpointing(self, models=None):
        if self.activation_checkpointing and self.activation_class_names is not None:
            models = models or self.models
            for model in models:
                cls_to_wrap = set()
                for class_name in self.activation_class_names:
                    for module in model.modules():
                        if module.__class__.__name__ == class_name:
                            cls_to_wrap.add(module.__class__)
                            break
                auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=cls_to_wrap)
                apply_activation_checkpointing(model, auto_wrap_policy=auto_wrap_policy)

    def prepare(self, dataloaders, models, optimizers, schedulers):
        self._dataloaders = utils.as_list(self.get_dataloaders(dataloaders))
        self._models = utils.as_list(self.get_models(models))
        self.set_ema_models()
        self.apply_activation_checkpointing()
        if self.distributed_type == DistributedType.FSDP:
            self._models = utils.as_list(self.accelerator.prepare(*self._models))
        self._optimizers = utils.as_list(self.get_optimizers(optimizers))
        self._schedulers = utils.as_list(self.get_schedulers(schedulers))
        if self.distributed_type == DistributedType.FSDP:
            objects = [self._dataloaders, self._optimizers, self._schedulers]
        else:
            objects = [self._dataloaders, self._models, self._optimizers, self._schedulers]
        inputs = functools.reduce(lambda x, y: x + y, objects)  # in case of multiple objects
        outputs = utils.as_list(self.accelerator.prepare(*inputs))
        start_idx = 0
        # in case of multiple objects
        for obj in objects:
            end_idx = start_idx + len(obj)
            obj[:] = outputs[start_idx:end_idx]
            start_idx = end_idx

    def train(self):
        self.print_before_train()
        dataloader_iter = iter(self.dataloader)
        for self._cur_step in range(self._cur_step, self.max_steps):
            self._cur_step += 1
            for _ in range(self.gradient_accumulation_steps):
                batch_dict = next(dataloader_iter)
                with self.accelerator.accumulate(*self.models):
                    losses = self.forward_step(batch_dict)
                    loss = self.parse_losses(losses)
                    self.backward_step(loss)
            self.print_step()
            self.save_checkpoint_step()
        self.print_after_train()
        self.accelerator.end_training()

    def forward_step(self, batch_dict):
        return self.model(batch_dict)

    def backward_step(self, loss):
        self.accelerator.backward(loss)
        max_grad_norm = self.kwargs.get('max_grad_norm', None)
        grad_norm_type = self.kwargs.get('grad_norm_type', 2)
        if self.accelerator.sync_gradients and max_grad_norm is not None:
            params = []
            for model in self.models:
                params += list(model.parameters())
            self.accelerator.clip_grad_norm_(params, max_grad_norm, grad_norm_type)
        for optimizer in self.optimizers:
            optimizer.step()
        for scheduler in self.schedulers:
            scheduler.step()
        for optimizer in self.optimizers:
            optimizer.zero_grad()
        if self.with_ema:
            for model, ema_model in zip(self.models, self.ema_models):
                model = self.accelerator.unwrap_model(model)
                ema_model.step(model.parameters())

    def save_checkpoint_step(self):
        if self._by_epoch and self.checkpoint_interval < self.max_epochs:
            checkpoint_interval = int(self.checkpoint_interval * self.epoch_size)
        else:
            checkpoint_interval = int(self.checkpoint_interval)
        if self.cur_step % checkpoint_interval == 0 or self.cur_step == self.max_steps:
            output_name = 'checkpoint_epoch_{}_step_{}'.format(self.cur_epoch, self.cur_step)
            output_dir = os.path.join(self.model_dir, output_name)
            if self.is_main_process:
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                self.remove_checkpoint()
            release_memory()
            self.accelerator.wait_for_everyone()
            self.accelerator.save_state(output_dir)

    def save_model_hook(self, models, weights, output_dir):
        if self.is_main_process:
            if len(weights) == 1:
                state_dict = weights.pop()
            elif len(weights) == 0 and self.distributed_type == DistributedType.DEEPSPEED:
                with torch.no_grad():
                    state_dict = self.accelerator.get_state_dict(models[0])
            else:
                assert False
            model = self.accelerator.unwrap_model(models[0])
            if isinstance(model, ModuleDict) or isinstance(model, nn.ModuleDict):
                model_names = list(model.keys())
                for model_name in model_names:
                    model_output_dir = os.path.join(output_dir, model_name)
                    os.makedirs(model_output_dir, exist_ok=True)
                    if hasattr(model[model_name], 'save_config'):
                        model[model_name].save_config(model_output_dir)
                    sub_state_dict = {
                        k[len(model_name) + 1 :]: v for k, v in state_dict.items() if k.startswith(model_name)
                    }
                    output_path = os.path.join(model_output_dir, WEIGHTS_NAME)
                    self.logger.info(f'Save {model_name} to {output_path}')
                    torch.save(sub_state_dict, output_path)
            else:
                model_name = getattr(self, 'model_name', 'model')
                model_output_dir = os.path.join(output_dir, model_name)
                os.makedirs(model_output_dir, exist_ok=True)
                if hasattr(model, 'save_config'):
                    model.save_config(model_output_dir)
                output_path = os.path.join(model_output_dir, WEIGHTS_NAME)
                self.logger.info(f'Save {model_name} to {output_path}')
                torch.save(state_dict, output_path)
            if self.with_ema:
                ema_model = self.ema_models[0]
                ema_model.store(model.parameters())
                ema_model.copy_to(model.parameters())
                state_dict = self.accelerator.get_state_dict(model, unwrap=False)
                if isinstance(model, ModuleDict) or isinstance(model, nn.ModuleDict):
                    model_names = list(model.keys())
                    for model_name in model_names:
                        model_output_dir = os.path.join(output_dir, model_name + '_ema')
                        os.makedirs(model_output_dir, exist_ok=True)
                        if hasattr(model[model_name], 'save_config'):
                            model[model_name].save_config(model_output_dir)
                        sub_state_dict = {
                            k[len(model_name) + 1 :]: v for k, v in state_dict.items() if k.startswith(model_name)
                        }
                        output_path = os.path.join(model_output_dir, WEIGHTS_NAME)
                        self.logger.info(f'Save {model_name}_ema to {output_path}')
                        torch.save(sub_state_dict, output_path)
                else:
                    model_name = getattr(self, 'model_name', 'model')
                    model_output_dir = os.path.join(output_dir, model_name + '_ema')
                    os.makedirs(model_output_dir, exist_ok=True)
                    if hasattr(model, 'save_config'):
                        model.save_config(model_output_dir)
                    output_path = os.path.join(model_output_dir, WEIGHTS_NAME)
                    self.logger.info(f'Save {model_name}_ema to {output_path}')
                    torch.save(state_dict, output_path)
                ema_model.restore(model.parameters())

    def load_model_hook(self, models, input_dir):
        if len(models) == 0:
            return
        assert len(models) == 1
        model = models.pop()
        if self.with_ema:
            ema_model = self.ema_models[0]
            if isinstance(model, ModuleDict) or isinstance(model, nn.ModuleDict):
                model_names = list(model.keys())
                state_dict = dict()
                for model_name in model_names:
                    input_path = os.path.join(input_dir, model_name + '_ema', WEIGHTS_NAME)
                    self.logger.info(f'Load {model_name}_ema from {input_path}')
                    sub_state_dict = torch.load(input_path, map_location='cpu')
                    sub_state_dict = {model_name + '.' + k: v for k, v in sub_state_dict.items()}
                    state_dict.update(sub_state_dict)
                model.load_state_dict(state_dict)
            else:
                model_name = getattr(self, 'model_name', 'model')
                input_path = os.path.join(input_dir, model_name + '_ema', WEIGHTS_NAME)
                self.logger.info(f'Load {model_name}_ema from {input_path}')
                state_dict = torch.load(input_path, map_location='cpu')
                model.load_state_dict(state_dict)
            ema_model.shadow_params = [p.clone().detach() for p in model.parameters()]
        if isinstance(model, ModuleDict) or isinstance(model, nn.ModuleDict):
            model_names = list(model.keys())
            state_dict = dict()
            for model_name in model_names:
                input_path = os.path.join(input_dir, model_name, WEIGHTS_NAME)
                self.logger.info(f'Load {model_name} from {input_path}')
                sub_state_dict = torch.load(input_path, map_location='cpu')
                sub_state_dict = {model_name + '.' + k: v for k, v in sub_state_dict.items()}
                state_dict.update(sub_state_dict)
            model.load_state_dict(state_dict)
        else:
            model_name = getattr(self, 'model_name', 'model')
            input_path = os.path.join(input_dir, model_name, WEIGHTS_NAME)
            self.logger.info(f'Load {model_name} from {input_path}')
            state_dict = torch.load(input_path, map_location='cpu')
            model.load_state_dict(state_dict)

    def print_before_train(self):
        if not self.is_main_process:
            return
        for model in self.models:
            self.logger.info(model)
        msg = 'num_processes: {}'.format(self.num_processes)
        msg += ', process_index: {}'.format(self.process_index)
        msg += ', data_size: {}'.format(self.data_size)
        msg += ', batch_size: {}'.format(self.batch_size)
        msg += ', epoch_size: {}'.format(self.epoch_size)
        self.logger.info(msg)
        self._epoch_tic = self._step_tic = self._start_tic = time.time()

    def print_step(self):
        if not self.is_main_process:
            return
        if self.cur_step % self.log_interval == 0:
            outputs = dict()
            for key, val in self._outputs.items():
                val = val['sum'] / val['num'] if val['num'] > 0 else float('nan')
                outputs[key] = val
            self._outputs.clear()
            self.accelerator.log(outputs, self.cur_step)
            time_cost = time.time() - self._step_tic
            self._step_tic = time.time()
            speed = self.log_interval * self.batch_size / time_cost
            eta_sec = max(0, time_cost / self.log_interval * (self.max_steps - self.cur_step))
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            lr = self.scheduler.get_last_lr()[0]
            if self._by_epoch:
                inner_step = (self.cur_step - 1) % self.epoch_size + 1
                msg = 'Epoch[%d/%d][%d/%d]' % (self.cur_epoch, self.max_epochs, inner_step, self.epoch_size)
            else:
                msg = 'Step[%d/%d]' % (self.cur_step, self.max_steps)
            msg += ' eta: %s, time: %.3f, speed: %.3f, lr: %.3e' % (eta_str, time_cost, speed, lr)
            if self.mixed_precision == 'fp16':
                if self.accelerator.scaler is not None and self.accelerator.scaler.is_enabled():
                    grad_scale = self.accelerator.scaler.get_scale()
                elif self.distributed_type == DistributedType.DEEPSPEED:
                    optimizer = self.optimizer
                    if hasattr(optimizer, 'optimizer'):
                        optimizer = optimizer.optimizer
                    if hasattr(optimizer, 'loss_scaler'):
                        grad_scale = optimizer.loss_scaler.cur_scale
                    elif hasattr(optimizer, 'cur_scale'):
                        grad_scale = optimizer.cur_scale
                    else:
                        assert False
                else:
                    grad_scale = None
            else:
                grad_scale = None
            if grad_scale is not None:
                msg += ', grad_scale: %.3f' % grad_scale
            for key, val in outputs.items():
                msg += ', %s: %.3f' % (key, val)
            self.logger.info(msg)
        if self._by_epoch and self.cur_step % self.epoch_size == 0:
            time_cost = time.time() - self._epoch_tic
            time_cost = str(datetime.timedelta(seconds=int(time_cost)))
            self._epoch_tic = time.time()
            self.logger.info('Total_time: %s' % time_cost)

    def print_after_train(self):
        if not self.is_main_process:
            return
        time_cost = time.time() - self._start_tic
        time_cost = str(datetime.timedelta(seconds=int(time_cost)))
        self.logger.info('Total_time: %s' % time_cost)

    def parse_losses(self, losses):
        if isinstance(losses, dict):
            assert 'total_loss' not in losses
            for key, val in losses.items():
                losses[key] = val.mean()
            loss = sum(losses.values())
            outputs = {'total_loss': loss}
            outputs.update(losses)
        elif isinstance(losses, torch.Tensor):
            loss = losses.mean()
            outputs = {'total_loss': loss}
        else:
            assert False
        if self.loss_nan_total_limit > 0 and torch.isnan(loss).any():
            self._loss_nan_count += 1
            if self._loss_nan_count > self.loss_nan_total_limit:
                exit(-1)
        else:
            self._loss_nan_count = 0
        for key, val in outputs.items():
            if key not in self._outputs:
                self._outputs[key] = {'sum': 0.0, 'num': 0}
            self._outputs[key]['sum'] += val.item()
            self._outputs[key]['num'] += 1
        return loss
