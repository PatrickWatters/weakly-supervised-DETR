import torch.multiprocessing as mp
import torch
import random
import time

def init_distributed_world(rank, world_size):
    import torch.distributed as dist
    backend = dist.Backend.NCCL if torch.cuda.is_available() else dist.Backend.GLOO  # type: ignore
    assert world_size <= torch.cuda.device_count()
    port = 20000
    init_method = 'tcp://localhost:{port}'.format(port=port)

    print('init process group')
    dist.init_process_group(init_method=init_method, backend=backend, rank=rank, world_size=world_size)
    print('I always get stuck here...')

    dist.all_reduce(torch.zeros([1]).cuda(), op=dist.ReduceOp.SUM)
    print('reduced', dist.get_rank())

def local_worker(rank, workers):
    init_distributed_world(rank, workers)

if __name__ == '__main__':
    workers = 2
    mp.spawn(local_worker, nprocs=workers, args=(workers, ), join=True)