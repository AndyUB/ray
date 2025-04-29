import fire
import time
import torch
import ray
from ray.experimental.channel.communicator import Communicator, TorchTensorAllocator
from ray.air._internal import torch_utils
from ray.experimental.util.types import ReduceOp
from typing import List, Optional, Tuple

from nccl import P2PActor, run_p2p


class TorchDistCommunicator(Communicator):
    """
    A custom NCCL group based on existing torch.distributed setup.
    """

    import cupy as cp

    def __init__(self, world_size, actor_handles):
        self._world_size = world_size
        self._actor_handles = actor_handles
        self._rank = None

    def initialize(self, rank: int) -> None:
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
        torch.distributed.send(value, peer_rank)

    def recv(
        self,
        shape: Tuple[int],
        dtype: "torch.dtype",
        peer_rank: int,
        allocator: Optional[TorchTensorAllocator] = None,
    ) -> "torch.Tensor":
        tensor = torch.empty(torch.Size(shape), dtype=dtype, device=self._device)
        torch.distributed.recv(tensor, peer_rank)
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
        raise NotImplementedError

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


def run_ray_p2p_with_torch_dist(
    num_iters: int = 1,
) -> None:
    ray.init()

    world_size = 2
    sender, receiver = P2PActor.remote(), P2PActor.remote()

    init_dist_start = time.perf_counter()
    refs = [
        sender.init_distributed.remote(world_size, 0),
        receiver.init_distributed.remote(world_size, 1),
    ]
    ray.wait(refs)
    init_dist_end = time.perf_counter()

    def log_elapse(start: float, end: float, event_name: str) -> str:
        elapse = (end - start) * 1e6
        print("[" + event_name + f"] {elapse:.2f} us")

    log_elapse(init_dist_start, init_dist_end, "init_dist")

    nccl_group = TorchDistCommunicator(world_size, [sender, receiver])
    run_p2p(
        num_iters=num_iters,
        actors=(sender, receiver),
        transport=nccl_group,
        init_ray=False,
    )

    ray.shutdown()


if __name__ == "__main__":
    fire.Fire(run_ray_p2p_with_torch_dist)
