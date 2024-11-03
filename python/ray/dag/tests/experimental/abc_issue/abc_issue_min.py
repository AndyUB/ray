import torch
import ray
from ray.experimental.collective import allreduce
from ray.dag import InputNode, MultiOutputNode
from typing import Tuple
from ray.air._internal import torch_utils


@ray.remote
class Worker:
    def __init__(self):
        self.device = torch_utils.get_devices()[0]

    @ray.method(num_returns=2)
    def return_two(self, size) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(size, device=self.device), torch.arange(
            size, device=self.device
        )


ray.init()
actor = Worker.options(num_gpus=1).remote()

# [TODO] Failed: No actor handle for class method output
with InputNode() as inp:
    tensor1, tensor2 = actor.return_two.bind(inp)
    tensor1 = allreduce.bind([tensor1])[0]
    dag = MultiOutputNode([tensor1, tensor2])
compiled_dag = dag.experimental_compile()
ref = compiled_dag.execute(10)
result = ray.get(ref)
print(result)
compiled_dag.teardown()

ray.shutdown()
