import logging
import os

os.environ["RAY_DEDUP_LOGS"] = "0"

import fire
import ray
import torch
import torch.distributed as dist

from common.time import get_time_perf_counter, secs_to_millis

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)


@ray.remote(num_gpus=1)
class Actor:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.process_group = None

    def init_process_group(self):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"

        logger.warning(f"Initializing process group for actor {self.rank}...")
        start_init = get_time_perf_counter()
        self.process_group = dist.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
        )
        end_init = get_time_perf_counter(sync=True)
        elapse_init_ms = secs_to_millis(end_init - start_init)
        logger.warning(
            f"Process group for actor {self.rank} initialized in {elapse_init_ms} ms"
        )

    def run_coll(self):
        logger.warning(f"Running coll for actor {self.rank}...")
        tensor = torch.zeros(1).to("cuda:0")
        tensor.fill_(self.rank)
        start_reduce = get_time_perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        end_reduce = get_time_perf_counter(sync=True)
        elapse_reduce_ms = secs_to_millis(end_reduce - start_reduce)
        logger.warning(
            f"Worker {self.rank} reduced tensor: {tensor} in {elapse_reduce_ms} ms"
        )

    def destroy_process_group(self):
        logger.warning(f"Destroying process group for actor {self.rank}...")
        dist.destroy_process_group()


def run_coll(
    devices: str = "1,2",
    world_size: int = 2,
):
    logger.info(f"Running with CUDA devices {devices}...")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    ray.init()

    actors = [Actor.remote(i, world_size) for i in range(world_size)]
    ray.get([actor.init_process_group.remote() for actor in actors])
    ray.get([actor.run_coll.remote() for actor in actors])
    ray.get([actor.destroy_process_group.remote() for actor in actors])

    ray.shutdown()


def run_coll_bench_w2():
    for devices in ["0,1", "1,2", "2,3", "3,4", "4,5", "5,6", "6,7"]:
        run_coll(devices, world_size=2)


def run_coll_bench_w4():
    for devices in ["0,1,2,3", "4,5,6,7"]:
        run_coll(devices, world_size=4)


def main(name: str):
    if name == "coll":
        run_coll()
    elif name == "coll_bench_w2":
        run_coll_bench_w2()
    elif name == "coll_bench_w4":
        run_coll_bench_w4()
    else:
        logger.error(f"Unknown name: {name}")


if __name__ == "__main__":
    fire.Fire(main)
