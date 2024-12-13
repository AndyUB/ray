import os

import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import ray
from ray.experimental.collective import allreduce
from ray.air._internal import torch_utils
from ray.dag import InputNode, MultiOutputNode

NUM_ITERS = 10
NUM_EXPRS = 10


@ray.remote(num_gpus=1)
class AllReduceWorker:
    def __init__(self):
        self.device = torch_utils.get_devices()[0]
        self.tensor = torch.ones(10, device=self.device) * 10

    def get_tensor(self, unused) -> torch.Tensor:
        return self.tensor

    def consume_tensors(self, *tensors: torch.Tensor) -> None:
        return None


def run_ray_allreduce() -> None:
    workers = [AllReduceWorker.remote() for _ in range(2)]
    with InputNode() as inp:
        tensors = [worker.get_tensor.bind(inp) for worker in workers]
        for _ in range(NUM_ITERS):
            tensors = allreduce.bind(tensors)
        ends = [worker.consume_tensors.bind(*tensors) for worker in workers]
        dag = MultiOutputNode(ends)

    compiled_dag = dag.experimental_compile()
    for i in range(NUM_EXPRS):
        start = time.perf_counter()
        ref = compiled_dag.execute(None)
        ray.get(ref)
        end = time.perf_counter()
        print(
            f"iteration {i} end-to-end time: {round((end - start) * 1e6)} us ({start, end})"
        )


def run_torch_allreduce():
    mp.spawn(run_torch_allreduce_per_process, nprocs=2)


def run_torch_allreduce_per_process(rank: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    dist.init_process_group("nccl", rank=rank, world_size=2)

    for it in range(NUM_EXPRS):
        tensor = torch.ones(10, device=f"cuda:{rank}") * 10
        times = []
        for _ in range(NUM_ITERS):
            start = time.perf_counter()
            dist.all_reduce(tensor)
            end = time.perf_counter()
            times.append((start, end))
        elapses = [end - start for start, end in times]
        for i, elapse in enumerate(elapses):
            print(f"iteration {it} torch allreduce {i}: {round(elapse * 1e6)} us")

    dist.destroy_process_group()


def main():
    run_ray_allreduce()
    run_torch_allreduce()


if __name__ == "__main__":
    main()
