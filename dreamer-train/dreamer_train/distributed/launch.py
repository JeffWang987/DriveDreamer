import os
import time
import socket
import subprocess

import torch
from accelerate import DistributedType
from accelerate.commands.config.config_args import ClusterConfig
from accelerate.utils import ComputeEnvironment

from ..configs import load_config
from ..utils import wait_for_gpu_memory
from .run_task import run_tasks


def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


class Launcher:
    def __init__(
        self,
        gpu_ids,
        num_machines=None,
        distributed_type=None,
        main_process_ip='127.0.0.1',
        main_process_port=None,
        num_cpu_threads_per_process=2,
        nccl_socket_ifname=None,
        save_config_path=None,
        save_hostfile_path=None,
        env=None,
        executable=None,
        until_completion=False,
        **kwargs,
    ):
        if num_machines is None:
            if isinstance(main_process_ip, str):
                num_machines = 1
            elif isinstance(main_process_ip, list):
                num_machines = len(main_process_ip)
            else:
                assert False
        if main_process_port is None:
            main_process_port = _find_free_port()
        cur_time = time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))
        os.makedirs('_tmp', exist_ok=True)
        if save_config_path is None:
            save_config_path = f'_tmp/{cur_time}_config.json'
        if save_hostfile_path is None:
            save_hostfile_path = f'_tmp/{cur_time}_hostfile'
        if env is None:
            env = os.environ
        if executable is None:
            process = subprocess.run(['which', 'accelerate'], env=env, capture_output=True, text=True)
            if process.returncode != 0:
                raise ValueError(process.stderr)
            executable = process.stdout.strip()
        if num_machines == 1:
            if distributed_type is None:
                distributed_type = DistributedType.MULTI_GPU
            distributed_type = DistributedType(distributed_type)
            if distributed_type != DistributedType.DEEPSPEED:
                kwargs['deepspeed_config'] = None
            num_processes = len(gpu_ids)
            gpu_ids = ','.join([str(i) for i in gpu_ids])
            cluster_config = ClusterConfig(
                compute_environment=ComputeEnvironment.LOCAL_MACHINE,
                distributed_type=distributed_type,
                mixed_precision=None,
                use_cpu=False,
                debug=False,
                num_processes=num_processes,
                gpu_ids=gpu_ids,
                main_process_ip=main_process_ip,
                main_process_port=main_process_port,
                **kwargs,
            )
        else:
            assert nccl_socket_ifname is not None
            env['NCCL_SOCKET_IFNAME'] = nccl_socket_ifname
            # env['NCCL_NET'] = 'SOCKET'
            # env['NCCL_IB_DISABLE'] = '1'
            env['NCCL_NET'] = 'IB'
            env['NCCL_IB_DISABLE'] = '0'
            env['NCCL_IB_PCI_RELAXED_ORDERING'] = '1'
            if distributed_type is None:
                distributed_type = DistributedType.DEEPSPEED
            distributed_type = DistributedType(distributed_type)
            assert distributed_type is DistributedType.DEEPSPEED
            if isinstance(gpu_ids[0], list):
                assert len(gpu_ids) == num_machines
            else:
                gpu_ids = [gpu_ids for _ in range(num_machines)]
            num_processes = sum(len(_) for _ in gpu_ids)
            assert isinstance(main_process_ip, list) and len(main_process_ip) == num_machines
            # update deepspeed_config
            with open(save_hostfile_path, 'w') as fn:
                for ip in main_process_ip:
                    fn.write(f'{ip} slots=8\n')
            includes = []
            for ip, gpu_ids_i in zip(main_process_ip, gpu_ids):
                gpu_ids_i = ','.join([str(i) for i in gpu_ids_i])
                includes.append(f'{ip}:{gpu_ids_i}')
            includes = '@'.join(includes)
            deepspeed_config = kwargs.pop('deepspeed_config')
            deepspeed_config.update(
                dict(
                    deepspeed_multinode_launcher='pdsh',
                    deepspeed_hostfile=save_hostfile_path,
                    deepspeed_inclusion_filter=includes,
                )
            )
            cluster_config = ClusterConfig(
                compute_environment=ComputeEnvironment.LOCAL_MACHINE,
                distributed_type=distributed_type,
                mixed_precision=None,
                use_cpu=False,
                debug=False,
                num_processes=num_processes,
                num_machines=num_machines,
                main_process_port=main_process_port,
                deepspeed_config=deepspeed_config,
                **kwargs,
            )
        cluster_config.to_json_file(save_config_path)
        self.cluster_config = cluster_config
        self.config_file = save_config_path
        self.hostfile_path = save_hostfile_path
        self.num_cpu_threads_per_process = num_cpu_threads_per_process
        self.env = env
        self.executable = executable
        self.until_completion = until_completion

    def launch(self, script):
        command = '{} launch'.format(self.executable)
        command += ' --config_file {}'.format(self.config_file)
        command += ' --num_cpu_threads_per_process {}'.format(self.num_cpu_threads_per_process)
        command += ' {}'.format(script)
        print(command)
        command = command.split(' ')
        try:
            while True:
                process = subprocess.run(command, env=self.env)
                if process.returncode != 0 and self.until_completion:
                    time.sleep(10)
                else:
                    break
        finally:
            os.remove(self.config_file)
            os.remove(self.hostfile_path)


def launch_from_config(config_path, runners, gpu_memory=None, seconds=10):
    config = load_config(config_path)
    gpu_ids = config.launch.gpu_ids
    num_machines = config.launch.get('num_machines', 1)
    if gpu_memory is not None:
        local_gpu_ids = gpu_ids[0] if isinstance(gpu_ids[0], list) else gpu_ids
        wait_for_gpu_memory(local_gpu_ids, gpu_memory, unit='MB', seconds=seconds)
    if num_machines == 1 and len(gpu_ids) == 1:
        torch.cuda.set_device(gpu_ids[0])
        run_tasks(config, runners)
    else:
        launcher = Launcher(**config.launch)
        file_path = os.path.join(os.path.abspath(__file__).split('launch')[0], 'run_task.py')
        launcher.launch('{} --config {} --runners {}'.format(file_path, config_path, runners))
