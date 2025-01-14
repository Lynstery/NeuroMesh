import os
import subprocess
import sys
import torch
from torch.distributed import init_process_group
from pathlib import Path
from crucio.autoencoder.util import home, PROJECT_DIR, YUV_ENABLED, CUDA_ENABLED

def check_world_size():
    total_gpu_num = int(subprocess.check_output(
        "nvidia-smi --list-gpus | wc -l", shell=True).decode().strip())
    visible_ids = [_ for _ in range(total_gpu_num)]
    #! Remove GPUs with error status
    err_gpuids = subprocess.check_output(
        "nvidia-smi --query-gpu=index,fan.speed --format=csv,noheader,nounits | awk -F, '$2 ~ /Error/{print $1}'", shell=True).decode().strip()
    if len(err_gpuids) > 0:
        err_gpuids = [int(_) for _ in err_gpuids.split('\n')]
        visible_ids = [_ for _ in visible_ids if _ not in err_gpuids]
    #! GPU failure may result in NAN gradient during training
    log_file = home+'/.gpu_log.txt'
    if os.path.exists(log_file):
        log_gpuids = []
        with open(log_file, 'r') as f:
            for line in f:
                log_gpuids.append(int(line.strip()))
        visible_ids = [_ for _ in visible_ids if _ not in log_gpuids]
    # Remove GPUs in busy status
    memory_usage = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F, '{printf \"%.2f\\n\", $2 / $1}'", shell=True).decode().strip()
    memory_usage = [float(_) for _ in memory_usage.split('\n')]
    busy_gpuids = [i for i, x in enumerate(memory_usage) if x > 0.5]
    visible_ids = [_ for _ in visible_ids if _ not in busy_gpuids]
    if sys.gettrace() is not None:
        visible_ids = [visible_ids[0]]
    # number of CUDA GPUs
    world_size = len(visible_ids)
    visible_str = list(map(str, visible_ids))
    visible_str = ', '.join(visible_str)
    if world_size > 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = visible_str
    elif world_size == 0:
        print('No CUDA-equipped GPU devices are available')
        exit(1)
    return world_size, visible_ids

WORLD_SIZE, visible_ids = check_world_size()

def get_gpu_model():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'], encoding='utf-8')
        gpu_model = result.strip().split('\n')[0]
        return gpu_model
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'Cannot get GPU model!'


def ddp_setup(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "13355"
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
    init_process_group(backend="nccl", rank=rank, world_size=WORLD_SIZE)
    torch.cuda.set_device(rank)

def print_device_info():
    print(f'[PROJECT_DIR={PROJECT_DIR}]')
    print(f'[Training using {WORLD_SIZE} GPU(s)]')
    if CUDA_ENABLED is True:
        print('[Evaluation using CUDA]')
        print(get_gpu_model())
        if torch.cuda.is_available() is False:
            print(
                'CUDA is not available due to mismatch between PyTorch version and CUDA version')
    else:
        print('[Evaluation using CPU]')
    if YUV_ENABLED is True:
        print('[Using YUV color space]')
    else:
        print('[Using RGB color space]')