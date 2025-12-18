import logging
import os

import fire
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from common.time import get_time_perf_counter, secs_to_millis

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)


def run_coll_worker(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    logger.info(f"Initializing process group for worker {rank}...")
    start_init = get_time_perf_counter()
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )
    end_init = get_time_perf_counter(sync=True)
    elapse_init_ms = secs_to_millis(end_init - start_init)
    logger.info(f"Process group for worker {rank} initialized in {elapse_init_ms} ms")

    tensor = torch.zeros(1).to(f"cuda:{rank}")
    tensor.fill_(rank)
    start_reduce = get_time_perf_counter()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    end_reduce = get_time_perf_counter(sync=True)
    elapse_reduce_ms = secs_to_millis(end_reduce - start_reduce)
    logger.info(f"Worker {rank} reduced tensor: {tensor} in {elapse_reduce_ms} ms")

    dist.destroy_process_group()


def run_coll():
    logger.info("Running coll...")
    world_size = 2
    mp.spawn(
        run_coll_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )


def run_coll_bench():
    for devices in ["0,1", "1,2", "2,3", "3,4", "4,5", "5,6", "6,7"]:
        logger.info(f"Running with CUDA devices {devices}...")
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        run_coll()


def main(name: str):
    if name == "coll":
        run_coll()
    elif name == "coll_bench":
        run_coll_bench()
    else:
        logger.error(f"Unknown name: {name}")


if __name__ == "__main__":
    fire.Fire(main)
