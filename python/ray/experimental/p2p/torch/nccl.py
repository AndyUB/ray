import fire
import os
import time
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import Any, List, Tuple


def run_p2p_worker(rank: int, num_iters: int, gpu_tensor: bool) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    init_start = time.perf_counter()
    dist.init_process_group(backend="nccl", rank=rank, world_size=2)
    init_end = time.perf_counter()
    cuda_visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    print(f"{cuda_visible_devices=}")

    if gpu_tensor:
        device = torch.device(f"cuda:{rank}")
    else:
        # Use CPU would error
        device = torch.device("cpu")
    size = 10

    iter_times: List[Tuple[float, float]] = []
    if rank == 0:
        for _ in range(num_iters):
            iter_start = time.perf_counter()
            send_buf = torch.zeros(size, device=device)
            print(f"{send_buf.device=}")
            dist.send(send_buf, dst=1)
            iter_end = time.perf_counter()
            iter_times.append((iter_start, iter_end))
    else:
        for _ in range(num_iters):
            iter_start = time.perf_counter()
            recv_buf = torch.empty(size, device=device)
            dist.recv(recv_buf, src=0)
            print(f"{recv_buf.device=}")
            iter_end = time.perf_counter()
            iter_times.append((iter_start, iter_end))

    teardown_start = time.perf_counter()
    dist.destroy_process_group()
    teardown_end = time.perf_counter()

    def log_elapse(start: float, end: float, event_name: str) -> str:
        elapse = (end - start) * 1e6
        print(f"<rank{rank}>[" + event_name + f"] {elapse:.2f} us")

    log_elapse(init_start, init_end, "init")
    for it, (start, end) in enumerate(iter_times):
        log_elapse(start, end, f"iter{it}")
    log_elapse(teardown_start, teardown_end, "teardown")


def run_p2p(
    num_iters: int = 1,
    cpu_tensor: bool = False,
) -> None:
    print(f"Running {num_iters} iters, cpu tensors: {cpu_tensor}")
    mp.spawn(
        run_p2p_worker,
        args=(num_iters, not cpu_tensor),
        nprocs=2,
        join=True,
    )


if __name__ == "__main__":
    fire.Fire(run_p2p)
