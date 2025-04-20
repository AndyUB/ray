import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run_torch_collective(rank: int, world_size: int, tensor_size: int) -> None:
    """
    A simple function to run a collective operation in PyTorch.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # Initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    # Create a tensor to be used in the collective operation
    tensor = torch.ones(tensor_size).cuda(rank) * (rank + 1)

    print(f"[before] Rank {rank} has tensor: {tensor}")

    # Perform an all-reduce operation
    dist.all_reduce(tensor)

    print(f"[after] Rank {rank} has tensor: {tensor}")

def main() -> None:
    world_size = 2
    tensor_size = 10
    mp.spawn(
        run_torch_collective,
        args=(world_size, tensor_size),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()