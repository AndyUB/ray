import os
import ray
from ray.air._internal import torch_utils
import torch

@ray.remote
class TorchTensorWorker:
    def __init__(self, rank: int):
        self.rank = rank
        assert torch.cuda.is_available(), f"{rank=} CUDA is not available"
        self.device = torch_utils.get_devices()[0]
        print(f"[init] Rank {rank} has device: {self.device}")

    def get_tensor(self, tensor_size: int) -> torch.Tensor:
        tensor = torch.ones(tensor_size).to(self.device) * (self.rank + 1)
        print(f"[tensor] Rank {self.rank} has tensor: {tensor}")
        return tensor

def ray_nccl_single() -> None:
    worker = TorchTensorWorker.options(num_gpus=1).remote(0)
    tensor = ray.get(worker.get_tensor.remote(10))
    print(f"[single] Rank 0 has tensor: {tensor}")

def ray_nccl(world_size: int, tensor_size: int) -> None:
    worker_cls = TorchTensorWorker.options(num_gpus=1)
    workers = [worker_cls.remote(rank) for rank in range(world_size)]
    tensors = ray.get([worker.get_tensor.remote(tensor_size) for worker in workers])

    for rank, tensor in enumerate(tensors):
        print(f"[before] Rank {rank} has tensor: {tensor}")

    # print(f"[after] Rank {rank} has tensor: {tensor}")

def main() -> None:
    world_size = 2
    tensor_size = 10
    ray.init()
    ray_nccl(world_size, tensor_size)
    ray.shutdown()

def main_single() -> None:
    ray.init()
    ray_nccl_single()
    ray.shutdown()

if __name__ == "__main__":
    # main()
    for i in reversed(range(8)):
        devices = str(i) + "," + str(i - 1)
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        print(f"CUDA_VISIBLE_DEVICES={devices} start")
        # main_single()
        main()
        print(f"CUDA_VISIBLE_DEVICES={devices} done")
