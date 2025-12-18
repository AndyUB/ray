import logging
import os

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

    def run_p2p(self):
        logger.warning(f"Running P2P for actor {self.rank}...")
        tensor = torch.zeros(1).to("cuda:0")

        if self.rank == 0:
            tensor.fill_(1.0)
            start_send = get_time_perf_counter()
            dist.send(tensor, dst=1)
            end_send = get_time_perf_counter(sync=True)
            elapse_send_ms = secs_to_millis(end_send - start_send)
            logger.warning(
                f"Actor {self.rank} sent tensor: {tensor} in {elapse_send_ms} ms"
            )

        else:
            start_recv = get_time_perf_counter()
            dist.recv(tensor, src=0)
            end_recv = get_time_perf_counter(sync=True)
            elapse_recv_ms = secs_to_millis(end_recv - start_recv)
            logger.warning(
                f"Actor {self.rank} received tensor: {tensor} in {elapse_recv_ms} ms"
            )

    def destroy_process_group(self):
        logger.warning(f"Destroying process group for actor {self.rank}...")
        dist.destroy_process_group()


def run_p2p():
    ray.init()

    actors = [Actor.remote(i, 2) for i in range(2)]
    ray.get([actor.init_process_group.remote() for actor in actors])
    ray.get([actor.run_p2p.remote() for actor in actors])
    ray.get([actor.destroy_process_group.remote() for actor in actors])

    ray.shutdown()


def run_p2p_bench():
    for devices in ["0,1", "1,2", "2,3", "3,4", "4,5", "5,6", "6,7"]:
        logger.info(f"Running with CUDA devices {devices}...")
        os.environ["CUDA_VISIBLE_DEVICES"] = devices
        run_p2p()


def main(name: str):
    if name == "p2p":
        run_p2p()
    elif name == "p2p_bench":
        run_p2p_bench()
    else:
        logger.error(f"Unknown name: {name}")


if __name__ == "__main__":
    fire.Fire(main)
