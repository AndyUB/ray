import torch
import time

import ray
from ray.dag import InputNode
from ray.air._internal import torch_utils
from ray.experimental.channel.torch_tensor_type import TorchTensorType


@ray.remote
class Worker:
    def __init__(self):
        self.device = torch_utils.get_devices()[0]

    def send(self, value: int):
        return torch.ones(100000, device=self.device) * value

    def recv(self, tensor: torch.Tensor):
        print(f"Worker.recv: {tensor}")
        return tensor


a = Worker.options(num_cpus=0, num_gpus=1).remote()
b = Worker.options(num_cpus=0, num_gpus=1).remote()

with InputNode() as inp:
    dag = a.send.bind(inp).with_type_hint(TorchTensorType(transport="nccl"))
    dag = b.recv.bind(dag)

compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)
ref = compiled_dag.execute(10)
result = ray.get(ref)
print(f"timestamp: {time.perf_counter()}, result: {result}")
