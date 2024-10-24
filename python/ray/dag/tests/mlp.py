from dataclasses import dataclass
import os
import time
import asyncio
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
import ray
from ray.dag import InputNode, MultiOutputNode
from ray.air._internal import torch_utils
from typing import List, Tuple, Optional

USE_GPU: bool = True
NUM_GPUS: int = 2
PRINT_DETAILS: bool = True
DISCARD_FIRST: bool = True


@dataclass
class Config:
    # Model config.
    input_size: int = 1000
    hidden_size: int = 2000
    output_size: int = 5
    # Inference config.
    batch_size: int = 100
    # Timing config.
    its: int = 100


CONFIG = Config()


class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        # First layer: Linear transformation from input to hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function (ReLU)
        self.relu = nn.ReLU()
        # Second layer: Linear transformation from hidden to output
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def generate_x() -> torch.Tensor:
    return torch.randn(CONFIG.batch_size, CONFIG.input_size)


def demo_basic() -> torch.Tensor:
    model = SimpleMLP(CONFIG.input_size, CONFIG.hidden_size, CONFIG.output_size)
    x = generate_x()
    output = model(x)
    print(output)
    return output


@ray.remote
class MLPActor:
    def __init__(self, use_gpu: bool = True):
        if use_gpu:
            self.device = torch_utils.get_devices()[0]
        else:
            self.device = "cpu"

        self.model = SimpleMLP(
            CONFIG.input_size, CONFIG.hidden_size, CONFIG.output_size
        ).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.model(x)


def demo_base_ray() -> float:
    if USE_GPU:
        actor_cls = MLPActor.options(num_gpus=1)
    else:
        actor_cls = MLPActor.options(num_gpus=0)
    actor = actor_cls.remote()
    x = generate_x()

    results: List[torch.Tensor] = []
    elapses: List[Tuple[float, float, float]] = []
    for _ in range(CONFIG.its):
        ref = actor.forward.bind(x)
        start = time.perf_counter()
        val = ref.execute()
        executed = time.perf_counter()
        result = ray.get(val)
        end = time.perf_counter()
        results.append(result)
        elapses.append((start, executed, end))

    check_results(results)
    avg = print_elapses(1, False, elapses)
    ray.kill(actor)
    return avg


def check_results(results: List[torch.Tensor]) -> None:
    if len(results) == 0:
        return
    result = results[0]
    for i in range(1, len(results)):
        assert torch.equal(results[i], result)


def print_elapses(
    num_dist: int,
    is_compiled: bool,
    elapses: List[Tuple[float, float, float]],
    description: Optional[str] = None,
    print_exec: bool = False,
) -> float:
    if PRINT_DETAILS:
        print_exec = True

    typ: str = "compiled graph" if is_compiled else "normal ray"
    if description:
        description = f"[{description}]"
    else:
        description = ""
    print(f"[MLP split on {num_dist} device(s)][{typ}]{description}")
    total = 0

    for i, (start, executed, end) in enumerate(elapses):
        if print_exec:
            print(
                f"#{i}: start={start}, executed={executed}(+{executed-start}), "
                f"end={end}(+{end-executed}), elapse={end-start}"
            )
        else:
            print(f"#{i}: start={start}, end={end}, elapse={end-start}")
        total += end - start

    avg = total / len(elapses)
    print(f"avg elapse: {avg}")
    if DISCARD_FIRST:
        assert len(elapses) > 1
        start, _, end = elapses[0]
        first = end - start
        print(f"first it: {first}")
        total -= first
        avg = total / (len(elapses) - 1)
        print(f"avg w/o 1st: {avg}")
    return avg


def demo_base_adag() -> float:
    if USE_GPU:
        actor_cls = MLPActor.options(num_gpus=1)
    else:
        actor_cls = MLPActor.options(num_gpus=0)
    actor = actor_cls.remote()

    with InputNode() as inp:
        dag = actor.forward.bind(inp)
    compiled_dag = dag.experimental_compile()

    x = generate_x()
    results: List[torch.Tensor] = []
    elapses: List[Tuple[float, float, float]] = []
    for _ in range(CONFIG.its):
        start = time.perf_counter()
        ref = compiled_dag.execute(x)
        executed = time.perf_counter()
        result = ray.get(ref)
        end = time.perf_counter()
        results.append(result)
        elapses.append((start, executed, end))

    check_results(results)
    avg = print_elapses(1, True, elapses)
    compiled_dag.teardown()
    ray.kill(actor)
    return avg


class MLPLayer1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPLayer1, self).__init__()
        # First layer: Linear transformation from input to hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function (ReLU)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass
        x = self.fc1(x)
        x = self.relu(x)
        return x


@ray.remote
class MLPLayer1Actor:
    def __init__(self, use_gpu: bool = True):
        if use_gpu:
            self.device = torch_utils.get_devices()[0]
        else:
            self.device = "cpu"

        self.model = MLPLayer1(CONFIG.input_size, CONFIG.hidden_size).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.model(x)


class MLPLayer2(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(MLPLayer2, self).__init__()
        # Second layer: Linear transformation from hidden to output
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass
        x = self.fc2(x)
        return x


@ray.remote
class MLPLayer2Actor:
    def __init__(self, use_gpu: bool = True):
        if use_gpu:
            self.device = torch_utils.get_devices()[0]
        else:
            self.device = "cpu"

        self.model = MLPLayer2(CONFIG.hidden_size, CONFIG.output_size).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.model(x)


def demo_pp_ray() -> float:
    if USE_GPU:
        actor1 = MLPLayer1Actor.options(num_gpus=1).remote()
        actor2 = MLPLayer2Actor.options(num_gpus=1).remote()
    else:
        actor1 = MLPLayer1Actor.options(num_gpus=0).remote()
        actor2 = MLPLayer2Actor.options(num_gpus=0).remote()

    x = generate_x()
    results: List[torch.Tensor] = []
    elapses: List[Tuple[float, float, float]] = []
    for _ in range(CONFIG.its):
        ref = actor1.forward.bind(x)
        ref = actor2.forward.bind(ref)
        start = time.perf_counter()
        val = ref.execute()
        executed = time.perf_counter()
        result = ray.get(val)
        end = time.perf_counter()
        results.append(result)
        elapses.append((start, executed, end))

    check_results(results)
    avg = print_elapses(2, False, elapses)
    ray.kill(actor1)
    ray.kill(actor2)
    return avg


def demo_pp_adag() -> float:
    if USE_GPU:
        actor1 = MLPLayer1Actor.options(num_gpus=1).remote()
        actor2 = MLPLayer2Actor.options(num_gpus=1).remote()
    else:
        actor1 = MLPLayer1Actor.options(num_gpus=0).remote()
        actor2 = MLPLayer2Actor.options(num_gpus=0).remote()

    with InputNode() as inp:
        dag = actor1.forward.bind(inp)
        dag = actor2.forward.bind(dag)
    compiled_dag = dag.experimental_compile()

    x = generate_x()
    results: List[torch.Tensor] = []
    elapses: List[Tuple[float, float, float]] = []
    for _ in range(CONFIG.its):
        start = time.perf_counter()
        ref = compiled_dag.execute(x)
        executed = time.perf_counter()
        result = ray.get(ref)
        end = time.perf_counter()
        results.append(result)
        elapses.append((start, executed, end))

    check_results(results)
    avg = print_elapses(2, True, elapses)
    compiled_dag.teardown()
    ray.kill(actor1)
    ray.kill(actor2)
    return avg


class MLPPart(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias):
        super(MLPPart, self).__init__()
        # First layer: Linear transformation from input to hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function (ReLU)
        self.relu = nn.ReLU()
        # Second layer: Linear transformation from hidden to output
        self.fc2 = nn.Linear(hidden_size, output_size, bias=False)
        self.bias = bias

    def forward(self, x):
        # Forward pass
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@ray.remote
class MLPPartActor:
    def __init__(
        self,
        num_parts: int,
        bias: torch.Tensor,
        torch_dist_rank: Optional[int] = None,
        use_gpu: bool = True,
    ):
        if use_gpu:
            self.device = torch_utils.get_devices()[0]
        else:
            self.device = "cpu"

        self.model = MLPPart(
            CONFIG.input_size,
            CONFIG.hidden_size // num_parts,
            CONFIG.output_size,
            bias.to(self.device),
        ).to(self.device)

        self.rank = torch_dist_rank
        self.world_size = num_parts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.model(x)

    def reduce(self, *x: torch.Tensor) -> torch.Tensor:
        assert len(x) > 0
        t = x[0].to(self.device)
        for i in range(1, len(x)):
            t = t + x[i].to(self.device)
        t = t + self.model.bias
        return t

    def torch_dist_init(self) -> None:
        # print("initing")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        # print("inited")

    def torch_dist_reduce(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        dist.reduce(x, 0)
        if self.rank == 0:
            x = x + self.bias
            return x
        else:
            return None

    def torch_dist_destroy(self) -> None:
        # print("destroying")
        dist.destroy_process_group()
        # print("destroyed")


def demo_tp_ray() -> float:
    num_actors = NUM_GPUS
    assert CONFIG.hidden_size % num_actors == 0

    h = CONFIG.hidden_size // num_actors
    # Calculate the bound for uniform distribution
    bound = 1 / (h**0.5)
    # Initialize a bias (size-1 vector) using the uniform distribution
    bias = torch.empty(CONFIG.output_size).uniform_(-bound, bound)

    if USE_GPU:
        actor_cls = MLPPartActor.options(num_gpus=1)
    else:
        actor_cls = MLPPartActor.options(num_gpus=0)
    actors = [actor_cls.remote(num_actors, bias) for _ in range(num_actors)]

    x = generate_x()
    results: List[torch.Tensor] = []
    elapses: List[Tuple[float, float, float]] = []
    for _ in range(CONFIG.its):
        refs = [actor.forward.bind(x) for actor in actors]
        ref = actors[0].reduce.bind(*refs)
        start = time.perf_counter()
        val = ref.execute()
        executed = time.perf_counter()
        result = ray.get(val)
        end = time.perf_counter()
        results.append(result)
        elapses.append((start, executed, end))

    check_results(results)
    avg = print_elapses(2, False, elapses)
    for actor in actors:
        ray.kill(actor)
    return avg


def demo_tp_adag() -> float:
    num_actors = NUM_GPUS
    assert CONFIG.hidden_size % num_actors == 0

    h = CONFIG.hidden_size // num_actors
    # Calculate the bound for uniform distribution
    bound = 1 / (h**0.5)
    # Initialize a bias (size-1 vector) using the uniform distribution
    bias = torch.empty(CONFIG.output_size).uniform_(-bound, bound)

    if USE_GPU:
        actor_cls = MLPPartActor.options(num_gpus=1)
    else:
        actor_cls = MLPPartActor.options(num_gpus=0)
    actors = [actor_cls.remote(num_actors, bias) for _ in range(num_actors)]

    with InputNode() as inp:
        fws = [actor.forward.bind(inp) for actor in actors]
        dag = actors[0].reduce.bind(*fws)
    compiled_dag = dag.experimental_compile()

    x = generate_x()
    results: List[torch.Tensor] = []
    elapses: List[Tuple[float, float, float]] = []
    for _ in range(CONFIG.its):
        start = time.perf_counter()
        ref = compiled_dag.execute(x)
        executed = time.perf_counter()
        result = ray.get(ref)
        end = time.perf_counter()
        results.append(result)
        elapses.append((start, executed, end))

    check_results(results)
    avg = print_elapses(2, True, elapses)
    compiled_dag.teardown()
    for actor in actors:
        ray.kill(actor)
    return avg


def demo_tp_dist() -> float:
    num_actors = NUM_GPUS
    assert CONFIG.hidden_size % num_actors == 0

    h = CONFIG.hidden_size // num_actors
    # Calculate the bound for uniform distribution
    bound = 1 / (h**0.5)
    # Initialize a bias (size-1 vector) using the uniform distribution
    bias = torch.empty(CONFIG.output_size).uniform_(-bound, bound)

    if USE_GPU:
        actor_cls = MLPPartActor.options(num_gpus=1)
    else:
        actor_cls = MLPPartActor.options(num_gpus=0)
    actors = [actor_cls.remote(num_actors, bias, i) for i in range(num_actors)]

    with InputNode() as inp:
        fws = [actor.forward.bind(inp) for actor in actors]
        reduces = [actor.torch_dist_reduce.bind(fw) for actor, fw in zip(actors, fws)]
        dag = MultiOutputNode(reduces)
    compiled_dag = dag.experimental_compile()

    x = generate_x()
    results: List[torch.Tensor] = []
    elapses: List[Tuple[float, float, float]] = []

    ray.get([actor.torch_dist_init.remote() for actor in actors])
    for _ in range(CONFIG.its):
        start = time.perf_counter()
        ref = compiled_dag.execute(x)
        executed = time.perf_counter()
        result = ray.get(ref)
        end = time.perf_counter()
        results.append(result[0])
        assert result[1:] == [None] * (num_actors - 1)
        elapses.append((start, executed, end))
    ray.get([actor.torch_dist_destroy.remote() for actor in actors])

    check_results(results)
    avg = print_elapses(2, True, elapses, description="torch dist")
    compiled_dag.teardown()
    for actor in actors:
        ray.kill(actor)
    return avg


def main() -> None:
    ray.init()
    if USE_GPU:
        assert (
            sum(node["Resources"].get("GPU", 0) for node in ray.nodes()) >= NUM_GPUS
        ), f"This test requires at least {NUM_GPUS} GPUs"

    # demo_basic()
    br = demo_base_ray()
    ba = demo_base_adag()
    pr = demo_pp_ray()
    pa = demo_pp_adag()
    tr = demo_tp_ray()
    ta = demo_tp_adag()
    td = demo_tp_dist()

    lats = {"br": br, "ba": ba, "pr": pr, "pa": pa, "tr": tr, "ta": ta, "td": td}
    for expr, lat in sorted(lats.items(), key=lambda x: x[1]):
        print(f"[{expr}] {lat}")
    ray.shutdown()


if __name__ == "__main__":
    main()
