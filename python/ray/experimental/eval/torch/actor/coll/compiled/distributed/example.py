import logging
import os

os.environ["RAY_DEDUP_LOGS"] = "0"

import time
from typing import List, Optional, Tuple

import fire
import ray
import ray.experimental.collective as collective
import torch
import torch.distributed as dist
from ray.air._internal import torch_utils
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.channel.communicator import Communicator, TorchTensorAllocator
from ray.experimental.util.types import ReduceOp

from common.time import get_time_perf_counter, secs_to_millis

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)


class TorchDistCommunicator(Communicator):
    import cupy as cp

    def __init__(self, world_size, actor_handles):
        self._world_size = world_size
        self._actor_handles = actor_handles
        self._rank = None

    def initialize(self, rank: int) -> None:
        logger.warning(f"Initializing communicator for rank {rank}...")
        expected_rank = self.get_rank(ray.get_runtime_context().current_actor)
        assert (
            rank == expected_rank
        ), f"NCCL actor's rank {rank} does not match expected rank {expected_rank}"
        self._rank = rank
        self._device = torch_utils.get_devices()[0]

    def get_rank(self, actor: ray.actor.ActorHandle) -> int:
        actor_ids = [a._ray_actor_id for a in self._actor_handles]
        try:
            rank = actor_ids.index(actor._ray_actor_id)
        except ValueError:
            raise ValueError("Actor is not in the NCCL group.")
        return rank

    def get_world_size(self) -> int:
        return self._world_size

    def get_self_rank(self) -> Optional[int]:
        return self._rank

    def get_actor_handles(self) -> List["ray.actor.ActorHandle"]:
        return self._actor_handles

    def send(self, value: "torch.Tensor", peer_rank: int) -> None:
        dist.send(value, peer_rank)

    def recv(
        self,
        shape: Tuple[int],
        dtype: "torch.dtype",
        peer_rank: int,
        allocator: Optional[TorchTensorAllocator] = None,
    ) -> "torch.Tensor":
        tensor = torch.empty(torch.Size(shape), dtype=dtype, device=self._device)
        dist.recv(tensor, peer_rank)
        return tensor

    def allgather(
        self,
        send_buf: "torch.Tensor",
        recv_buf: "torch.Tensor",
    ) -> None:
        raise NotImplementedError

    def allreduce(
        self,
        send_buf: "torch.Tensor",
        recv_buf: "torch.Tensor",
        op: ReduceOp = ReduceOp.SUM,
    ) -> None:
        # [TODO] Use this for ray dag coll api.
        dist.all_reduce(send_buf)
        recv_buf.copy_(send_buf)

    def reducescatter(
        self,
        send_buf: "torch.Tensor",
        recv_buf: "torch.Tensor",
        op: ReduceOp = ReduceOp.SUM,
    ) -> None:
        raise NotImplementedError

    @property
    def recv_stream(self) -> Optional["cp.cuda.ExternalStream"]:
        import cupy as cp

        return cp.cuda.get_current_stream()

    @property
    def send_stream(self) -> Optional["cp.cuda.ExternalStream"]:
        import cupy as cp

        return cp.cuda.get_current_stream()

    @property
    def coll_stream(self) -> Optional["cp.cuda.ExternalStream"]:
        import cupy as cp

        return cp.cuda.get_current_stream()

    def destroy(self) -> None:
        pass

    def get_transport_name(self) -> str:
        return "nccl"


@ray.remote(num_gpus=1)
class Actor:
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.start = get_time_perf_counter()

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

    def run_send(self, _):
        tensor = torch.zeros(1).to("cuda:0")
        tensor.fill_(self.rank)
        return tensor

    def run_coll(self, tensor):
        logger.warning(f"Running coll for actor {self.rank}...")
        start_reduce = get_time_perf_counter()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        end_reduce = get_time_perf_counter(sync=True)
        elapse_reduce_ms = secs_to_millis(end_reduce - start_reduce)
        logger.warning(
            f"Worker {self.rank} reduced tensor: {tensor} in {elapse_reduce_ms} ms"
        )

    def get_time(self):
        self.end = get_time_perf_counter(sync=True)
        elapse_ms = secs_to_millis(self.end - self.start)
        logger.warning(f"Actor {self.rank} completed in {elapse_ms} ms")
        time.sleep(1)


def run_coll(
    devices: str = "1,2",
    world_size: int = 2,
):
    logger.info(f"Running with CUDA devices {devices}...")
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    ray.init()

    actors = [Actor.remote(i, world_size) for i in range(world_size)]
    ray.get([actor.init_process_group.remote() for actor in actors], timeout=120)
    communicator = TorchDistCommunicator(world_size, actors)

    with InputNode() as inp:
        tensors = [actor.run_send.bind(inp) for actor in actors]
        coll_tensors = collective.allreduce.bind(tensors, transport=communicator)
        dag = MultiOutputNode(coll_tensors)

    compiled_dag = dag.experimental_compile(_default_communicator=communicator)

    ray.get(compiled_dag.execute(None), timeout=120)
    ray.get([actor.get_time.remote() for actor in actors], timeout=120)

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
