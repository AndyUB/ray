import torch
import ray
import time
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
        timestamp = time.perf_counter()
        print(f"{timestamp}: returning two")
        return torch.ones(size, device=self.device), torch.arange(
            size, device=self.device
        )

    @ray.method(num_returns=1)
    def return_one(self, size) -> torch.Tensor:
        return torch.ones(size, device=self.device)


ray.init()
num_workers = 2
actors = [Worker.options(num_gpus=1).remote() for _ in range(num_workers)]

# with InputNode() as inp:
#     dag = actors[0].return_one.bind(inp)

# compiled_dag = dag.experimental_compile()
# ref = compiled_dag.execute(10)
# result = ray.get(ref)
# print(result)
# compiled_dag.teardown()

# with InputNode() as inp:
#     dag = actors[0].return_one.bind(inp)
#     dag = allreduce.bind([dag])[0]
# compiled_dag = dag.experimental_compile()
# ref = compiled_dag.execute(10)
# result = ray.get(ref)
# print(result)
# compiled_dag.teardown()

# [TODO] Failed: No actor handle for class method output
# with InputNode() as inp:
#     tensor1, tensor2 = actors[0].return_two.bind(inp)
#     tensor1 = allreduce.bind([tensor1])[0]
#     dag = MultiOutputNode([tensor1, tensor2])
# compiled_dag = dag.experimental_compile()
# ref = compiled_dag.execute(10)
# result = ray.get(ref)
# print(result)
# compiled_dag.teardown()

# with InputNode() as inp:
#     tensor1 = actors[0].return_one.bind(inp)
#     tensor2 = actors[1].return_one.bind(inp)
#     tensors = allreduce.bind([tensor1, tensor2])
#     dag = MultiOutputNode(tensors)
# compiled_dag = dag.experimental_compile()
# ref = compiled_dag.execute(10)
# result = ray.get(ref)
# print(result)
# compiled_dag.teardown()

# [TODO] Failed: No actor handle for class method output
# with InputNode() as inp:
#     tensor1, tensor3 = actors[0].return_two.bind(inp)
#     tensor2 = actors[1].return_one.bind(inp)
#     tensors = allreduce.bind([tensor1, tensor2])
#     dag = MultiOutputNode(tensors + [tensor3])
# compiled_dag = dag.experimental_compile()
# ref = compiled_dag.execute(10)
# result = ray.get(ref)
# print(result)
# compiled_dag.teardown()

# [TODO] Failed: No actor handle for class method output
# with InputNode() as inp:
#     tensor1, tensor2 = actors[0].return_two.bind(inp)
#     tensor3, tensor4 = actors[1].return_two.bind(inp)
#     tensors = allreduce.bind([tensor1, tensor3])
#     dag = MultiOutputNode(tensors + [tensor2, tensor4])
# compiled_dag = dag.experimental_compile()
# ref = compiled_dag.execute(10)
# result = ray.get(ref)
# print(result)
# compiled_dag.teardown()

# [TODO] Failed: Deadlock
# with InputNode() as inp:
#     tensor1, tensor2 = actors[0].return_two.bind(inp)
#     tensor3, tensor4 = actors[1].return_two.bind(inp)
#     allreduce1 = allreduce.bind([tensor1, tensor4])
#     dag = MultiOutputNode(allreduce1 + [tensor2, tensor3])
# compiled_dag = dag.experimental_compile()
# ref = compiled_dag.execute(10)
# result = ray.get(ref)
# print(result)
# compiled_dag.teardown()

# [TODO] Failed: Deadlock
# with InputNode() as inp:
#     tensor1, tensor2 = actors[0].return_two.bind(inp)
#     tensor3, tensor4 = actors[1].return_two.bind(inp)
#     allreduce1 = allreduce.bind([tensor2, tensor3])
#     dag = MultiOutputNode(allreduce1 + [tensor1, tensor4])
# compiled_dag = dag.experimental_compile()
# ref = compiled_dag.execute(10)
# result = ray.get(ref)
# print(result)
# compiled_dag.teardown()

# [TODO] Failed: No actor handle for class method output
# with InputNode() as inp:
#     tensor1, tensor2 = actors[0].return_two.bind(inp)
#     tensor3, tensor4 = actors[1].return_two.bind(inp)
#     allreduce1 = allreduce.bind([tensor2, tensor4])
#     dag = MultiOutputNode(allreduce1 + [tensor1, tensor3])
# compiled_dag = dag.experimental_compile()
# ref = compiled_dag.execute(10)
# result = ray.get(ref)
# print(result)
# compiled_dag.teardown()

# [TODO] Failed: No actor handle for class method output
# with InputNode() as inp:
#     tensor1, tensor2 = actors[0].return_two.bind(inp)
#     tensor3, tensor4 = actors[1].return_two.bind(inp)
#     allreduce1 = allreduce.bind([tensor1, tensor3])
#     allreduce2 = allreduce.bind([tensor2, tensor4])
#     dag = MultiOutputNode(allreduce1 + allreduce2)
# compiled_dag = dag.experimental_compile()
# ref = compiled_dag.execute(10)
# result = ray.get(ref)
# print(result)
# compiled_dag.teardown()

# [TODO] Failed: Deadlock
# with InputNode() as inp:
#     tensor1, tensor2 = actors[0].return_two.bind(inp)
#     tensor3, tensor4 = actors[1].return_two.bind(inp)
#     allreduce1 = allreduce.bind([tensor1, tensor4])
#     allreduce2 = allreduce.bind([tensor2, tensor3])
#     dag = MultiOutputNode(allreduce1 + allreduce2)
# compiled_dag = dag.experimental_compile()
# ref = compiled_dag.execute(10)
# result = ray.get(ref)
# print(result)
# compiled_dag.teardown()

ray.shutdown()
