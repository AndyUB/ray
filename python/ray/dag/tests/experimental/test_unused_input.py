import ray
from ray.dag import InputNode
import torch
from ray.experimental.channel.torch_tensor_nccl_channel import TorchTensorType

@ray.remote(num_gpus=1)
class Actor:
    def __init__(self) -> None:
        pass

    def test(self, tensor: torch.Tensor):
        return type(tensor)


if __name__ == "__main__":
    ray.init()
    actor = Actor.remote()

    with InputNode() as dag_input:
        input1, input2 = dag_input[0], dag_input[1]
        input1 = input1.with_type_hint(TorchTensorType())
        dag = actor.test.bind(input1)

    adag = dag.experimental_compile()
    output = ray.get(
        adag.execute([torch.randn(2, 16), 1.0])
    )  # This always outputs list, should be torch.Tensor?
    print(output)

    adag.teardown()
