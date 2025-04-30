import fire
import os
import time
import torch
import ray
from ray.dag import InputNode
from ray.experimental.channel.communicator import Communicator
from ray.air._internal import torch_utils

from typing import Any, List, Tuple, Union


@ray.remote(num_gpus=1)
class P2PActor:
    def __init__(self, use_cpu: bool = False):
        if use_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch_utils.get_devices()[0]
        self.data = torch.zeros(10, device=self.device)

    def get_tensor(self, _dependency: Any) -> torch.Tensor:
        return self.data

    def recv_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.device == self.device
        return tensor

    def init_distributed(self, world_size, rank):
        print(os.environ["CUDA_VISIBLE_DEVICES"])
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size, rank=rank
        )
        # [NOTE] Does not work
        # torch.cuda.set_device(rank)
        # torch.cuda.synchronize()
        # torch.distributed.barrier()
        print(f"rank{rank} inited")
        # size = 10
        # start = time.perf_counter()
        # if rank == 0:
        #     send_buf = torch.zeros(size, device=self.device)
        #     print(f"{send_buf.device=}")
        #     # print(f"sender sleeping")
        #     # time.sleep(3)
        #     torch.distributed.send(send_buf, dst=1)
        #     print(f"sent")
        # else:
        #     recv_buf = torch.empty(size, device=self.device)
        #     # print(f"recver sleeping")
        #     # time.sleep(6)
        #     torch.distributed.recv(recv_buf, src=0)
        #     print(f"{recv_buf.device=}")
        #     print(f"recved")
        # end = time.perf_counter()
        # elapse = (end - start) * 1e3
        # print(f"elapse: {elapse} ms")


def run_p2p(
    num_iters: int = 1,
    init_ray: bool = True,
    actors: Tuple[ray.actor.ActorHandle, ray.actor.ActorHandle] = (None, None),
    transport: Union[str, Communicator] = "nccl",
    timeout: float = 10,
    cpu_tensor: bool = False,
) -> None:
    if init_ray:
        ray.init()

    sender, receiver = actors
    cpu_tensor = bool(cpu_tensor)
    print(f"ray+cupy, {cpu_tensor=}")
    if sender is None:
        sender = P2PActor.remote(cpu_tensor)
    if receiver is None:
        receiver = P2PActor.remote(cpu_tensor)
    with InputNode() as inp:
        tensor = sender.get_tensor.bind(inp).with_tensor_transport(transport=transport)
        dag = receiver.recv_tensor.bind(tensor)
    compile_start = time.perf_counter()
    compiled_dag = dag.experimental_compile()
    compile_end = time.perf_counter()
    iter_times: List[Tuple[float, float]] = []
    for _ in range(num_iters):
        iter_start = time.perf_counter()
        ref = compiled_dag.execute(None)
        _result = ray.get(ref)
        iter_end = time.perf_counter()
        iter_times.append((iter_start, iter_end))
    teardown_start = time.perf_counter()
    compiled_dag.teardown()
    teardown_end = time.perf_counter()

    def log_elapse(start: float, end: float, event_name: str) -> str:
        elapse = (end - start) * 1e6
        print("[" + event_name + f"] {elapse:.2f} us")

    log_elapse(compile_start, compile_end, "compile")
    for it, (start, end) in enumerate(iter_times):
        log_elapse(start, end, f"iter{it}")
    log_elapse(teardown_start, teardown_end, "teardown")

    if init_ray:
        ray.shutdown()


if __name__ == "__main__":
    fire.Fire(run_p2p)
