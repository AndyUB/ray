import logging
import os
import sys
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Tuple, Type, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import ray
from ray.air._internal import torch_utils
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce


logger = logging.getLogger(__name__)


@dataclass
class Config:
    "Configuration for the demo model."
    # Model config.
    num_layers: int = 2
    layer_size: int = 10  # The layer is a square.
    # Training config.
    dtype: torch.dtype = torch.float32
    it: int = 1
    lr: int = 1e-3
    # Distributed config.
    num_actors: int = 2


CONFIG = Config()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 42
set_seed(SEED)


class DDPModel(ABC):
    def __init__(self, num_layers: int):
        self._num_layers = num_layers
        self._device = torch_utils.get_devices()[0]

    @property
    def num_layers(self) -> int:
        return self._num_layers

    def train(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        input = self.start_train(X)
        for i in range(self.num_layers):
            input = self.forward(i, input)
        pred = input
        grad = self.loss(pred, Y)
        updates = []
        for i in reversed(range(self.num_layers)):
            grad, grad_update = self.backward(i, grad)
            updates.append(self.update(i, grad_update))
        return self.finish_train(*updates)

    @abstractmethod
    def start_train(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, layer_idx: int, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss(self, pred: torch.Tensor, Y: torch.Tensor) -> Tuple[torch.Tensor, None]:
        raise NotImplementedError

    @abstractmethod
    def backward(
        self, layer_idx: int, grad: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def update(self, layer_idx: int, grad: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def finish_train(self, *updates: torch.Tensor) -> List[torch.Tensor]:
        raise NotImplementedError

    def get_grad_to_reduce(
        self, grad: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        _, reduce_grad = grad
        return reduce_grad


class Model(torch.nn.Module):
    def __init__(
        self, layer_size: int, num_layers: int, device: torch.device, dtype: torch.dtype
    ):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        super(Model, self).__init__()

        self.layers: List[torch.nn.Module] = []
        for _ in range(num_layers):
            self.layers.append(
                torch.nn.Linear(
                    layer_size, layer_size, device=device, dtype=dtype, bias=False
                )
            )
            self.layers.append(torch.nn.ReLU())
        self.layers: nn.ModuleList = nn.ModuleList(self.layers)
        self.inputs: List[torch.Tensor] = []
        self.outputs: List[torch.Tensor] = []
        self.activations: List[torch.Tensor] = []
        self.lr: float = CONFIG.lr
        self.criterion = nn.MSELoss()
        self.optimizers: List[optim.SGD] = [
            optim.SGD(self.layers[2 * i].parameters(), lr=self.lr)
            for i in range(num_layers)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        self.inputs.append(x)
        linear_layer: torch.nn.Module = self.layers[2 * layer_idx]
        y: torch.Tensor = linear_layer(x)
        relu_activation: torch.nn.Module = self.layers[2 * layer_idx + 1]
        z: torch.Tensor = relu_activation(y)
        self.activations.append(z)
        return z

    def backward_layer(
        self, grad: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z: torch.Tensor = self.activations[layer_idx]
        x: torch.Tensor = self.inputs[layer_idx]
        layer: torch.nn.Linear = self.layers[2 * layer_idx]
        W: torch.Tensor = layer.weight
        optimizer = self.optimizers[layer_idx]
        optimizer.zero_grad()
        z.backward(gradient=torch.ones_like(z), retain_graph=True, inputs=[W, x])
        return x.grad * grad, W.grad * grad

    def update_layer(self, grad: torch.Tensor, layer_idx: int) -> torch.Tensor:
        layer: torch.nn.Linear = self.layers[2 * layer_idx]
        [param for param in layer.parameters()][0].grad = grad
        optimizer = self.optimizers[layer_idx]
        optimizer.step()
        # with torch.no_grad():
        #     layer.weight -= self.lr * grad
        return layer.weight


@ray.remote
class TorchDDPModel(DDPModel):
    def __init__(self, num_layers: int, layer_size: int):
        super().__init__(num_layers)

        self._model: Model = Model(layer_size, num_layers, self._device, torch.float32)

    def start_train(self, x: torch.Tensor) -> torch.Tensor:
        # self._model.zero_grad()
        return x.to(self._device)

    def forward(self, layer_idx: int, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self._device)
        return self._model.forward_layer(input, layer_idx)

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, None]:
        y = y.to(self._device)
        pred = pred.to(self._device)
        loss: torch.Tensor = self._model.criterion(pred, y)
        loss.backward(retain_graph=True, inputs=[pred])
        return pred.grad, None

    def backward(
        self, layer_idx: int, grad: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bp_grad, _ = grad
        bp_grad = bp_grad.to(self._device)
        return self._model.backward_layer(bp_grad, layer_idx)

    def update(self, layer_idx: int, grad: torch.Tensor) -> torch.Tensor:
        grad = grad.to(self._device)
        return self._model.update_layer(grad, layer_idx)

    def finish_train(self, *updates: torch.Tensor) -> List[torch.Tensor]:
        return updates


def generate_x_y(config: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    layer_size = config.layer_size
    num_actors = config.num_actors
    dtype = config.dtype

    shape = (num_actors * layer_size, layer_size)
    numel = shape[0] * shape[1]

    x = torch.arange(numel, dtype=dtype, requires_grad=True) * 0.1
    x = x.reshape(shape)
    y = torch.arange(numel, dtype=dtype) * 10
    y = y.reshape(shape)

    return x, y


def run_experiment(model: Type[DDPModel]) -> None:
    actor_cls = model.options(num_gpus=1)
    num_layers, layer_size = CONFIG.num_layers, CONFIG.layer_size
    num_actors = CONFIG.num_actors
    actors = [actor_cls.remote(num_layers, layer_size) for _ in range(num_actors)]

    x, y = generate_x_y(CONFIG)
    xs = torch.tensor_split(x, num_actors)
    ys = torch.tensor_split(y, num_actors)

    with InputNode() as inp:
        losses = []
        for i, actor in enumerate(actors):
            x = inp[i]
            y = inp[num_actors + i]
            start = actor.start_train.bind(x)
            forwards = [start]
            for j in range(num_layers):
                forwards.append(actor.forward.bind(j, forwards[-1]))
            loss = actor.loss.bind(forwards[-1], y)
            losses.append(loss)
        output = []
        grads = losses
        for j in reversed(range(num_layers)):
            # grads_to_reduce = []
            for i, actor in enumerate(actors):
                # grads[i], grad_to_reduce = actor.backward.bind(j, grads[i])
                # grads_to_reduce.append(grad_to_reduce)
                grads[i] = actor.backward.bind(j, grads[i])
            reduced_grads = allreduce.bind(
                [
                    actor.get_grad_to_reduce.bind(grads[i])
                    for i, actor in enumerate(actors)
                ]
            )
            # reduced_grads = allreduce.bind(grads_to_reduce)
            updates = [
                actor.update.bind(j, reduced_grad)
                for actor, reduced_grad in zip(actors, reduced_grads)
            ]
            output.append(updates)
        ends = [
            actor.finish_train.bind(
                *[output[j][i] for j in reversed(range(num_layers))]
            )
            for i, actor in enumerate(actors)
        ]
        dag = MultiOutputNode(ends)

    compiled_dag = dag.experimental_compile()
    it = CONFIG.it
    for _ in range(it):
        ref = compiled_dag.execute(*xs, *ys)
        result = ray.get(ref)
        print(f"[ray]: {result}")

    compiled_dag.teardown()

    for actor in actors:
        ray.kill(actor)


def check_torch_model_correctness() -> None:
    check_torch_model_correctness_auto()
    check_torch_model_correctness_dist()


def check_torch_model_correctness_auto() -> None:
    x, y = generate_x_y(CONFIG)
    device = "cuda:0"
    x = x.to(device)
    y = y.to(device)
    model = Model(CONFIG.layer_size, CONFIG.num_layers, device, CONFIG.dtype)
    criterion = model.criterion
    optimizer = optim.SGD(model.parameters(), lr=model.lr)

    for _ in range(CONFIG.it):
        optimizer.zero_grad()
        pred: torch.Tensor = model(x)
        loss: torch.Tensor = criterion(pred, y)
        loss.backward()
        optimizer.step()

    for i in range(0, len(model.layers), 2):
        layer: torch.nn.Linear = model.layers[i]
        print(f"[auto] layer {i // 2}: {layer.weight}")


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    # create model and move it to GPU with id rank
    model = Model(
        CONFIG.layer_size, CONFIG.num_layers, f"cuda:{rank}", CONFIG.dtype
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = model.criterion
    optimizer = optim.SGD(ddp_model.parameters(), lr=model.lr)
    optimizer.zero_grad()

    num_actors = CONFIG.num_actors
    x, y = generate_x_y(CONFIG)
    xs = torch.tensor_split(x, num_actors)
    ys = torch.tensor_split(y, num_actors)
    x = xs[rank]
    y = ys[rank]
    x = x.to(rank)
    y = y.to(rank)

    outputs = ddp_model(x)
    labels = y
    loss_fn(outputs, labels).backward()
    optimizer.step()

    for i in range(0, len(model.layers), 2):
        layer: torch.nn.Linear = model.layers[i]
        print(f"[dist] layer {i // 2}: {layer.weight}")

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn, args=(world_size,), nprocs=world_size, join=True)


def check_torch_model_correctness_dist():
    n_gpus = torch.cuda.device_count()
    assert (
        n_gpus >= CONFIG.num_actors
    ), f"Requires at least {CONFIG.num_actors} GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(demo_basic, world_size)


def main() -> None:
    ray.init()
    if sum(node["Resources"].get("GPU", 0) for node in ray.nodes()) < CONFIG.num_actors:
        print(f"Needs at least {CONFIG.num_actors} GPUs")
        return

    run_experiment(TorchDDPModel)

    ray.shutdown()

    check_torch_model_correctness()


if __name__ == "__main__":
    main()

# 0. cleanup
# 1. baseline (identify performance overhead)
# 2. multiple steps