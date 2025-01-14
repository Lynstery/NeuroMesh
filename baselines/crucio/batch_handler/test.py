import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from crucio.autoencoder.util import WORLD_SIZE, ddp_setup, print_device_info

class dff_round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # 确保结果在与输入相同的设备上
        return torch.round(input).to(input.device)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) 

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    x = torch.tensor([1.2, 2.5, 3.7], requires_grad=True, device=f'cuda:{rank}')

    print(f"[rank={rank}] x: {x.device}")
    y = dff_round.apply(x)
    print(f"[rank={rank}] y.device: {y.device}")

    # 模拟损失计算和反向传播
    loss = y.sum()
    loss.backward()
    print(f"[rank={rank}] x.grad: {x.grad}")

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size, join=True)