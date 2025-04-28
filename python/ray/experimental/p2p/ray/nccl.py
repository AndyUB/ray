import fire
import time
import torch
import ray
from ray.dag import InputNode

from typing import Any, List, Tuple


@ray.remote(num_gpus=1)
class P2PActor:
    def __init__(self):
        self.data = torch.zeros(10)

    def get_tensor(self, _dependency: Any) -> torch.Tensor:
        return self.data

    def recv_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def run_p2p(
    num_iters: int = 1,
) -> None:
    ray.init()

    sender, receiver = P2PActor.remote(), P2PActor.remote()
    with InputNode() as inp:
        tensor = sender.get_tensor.bind(inp).with_tensor_transport(transport="nccl")
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

    ray.shutdown()


if __name__ == "__main__":
    fire.Fire(run_p2p)
