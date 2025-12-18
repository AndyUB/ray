import logging
import os

os.environ["RAY_DEDUP_LOGS"] = "0"

import time

import fire
import ray
import torch
from ray.dag import InputNode

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
        self.start = get_time_perf_counter()

    def run_send(self, _):
        return 0
        tensor = torch.zeros(1).to("cuda:0")
        tensor.fill_(1.0)
        self.end = get_time_perf_counter(sync=True)
        return tensor

    def run_recv(self, tensor: torch.Tensor):
        return 0
        tensor += 1
        self.end = get_time_perf_counter(sync=True)
        return tensor

    def get_time(self):
        elapse_ms = secs_to_millis(self.end - self.start)
        logger.warning(f"Actor {self.rank} completed in {elapse_ms} ms")
        time.sleep(1)


def run_p2p(devices: str = "1,2"):
    logger.info(f"Running with CUDA devices {devices}...")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    ray.init()

    actors = [Actor.remote(i, 2) for i in range(2)]

    with InputNode() as inp:
        dag = actors[0].run_send.bind(inp).with_tensor_transport(transport="nccl")
        dag = actors[1].run_recv.bind(dag)
        # dag = actors[1].run_send.bind(dag).with_tensor_transport(transport="nccl")
        # dag = actors[0].run_recv.bind(dag)

    compiled_dag = dag.experimental_compile()

    ray.get(compiled_dag.execute(None), timeout=120)
    # ray.get([actor.get_time.remote() for actor in actors], timeout=120)

    ray.shutdown()


def run_p2p_bench():
    for devices in ["0,1", "1,2", "2,3", "3,4", "4,5", "5,6", "6,7"]:
        run_p2p(devices)


def main(name: str):
    if name == "p2p":
        run_p2p()
    elif name == "p2p_bench":
        run_p2p_bench()
    else:
        logger.error(f"Unknown name: {name}")


if __name__ == "__main__":
    fire.Fire(main)
