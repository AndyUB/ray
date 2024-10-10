import logging
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Tuple, Type, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import ray
from ray.air._internal import torch_utils
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce


logger = logging.getLogger(__name__)

USE_GPU = bool(os.environ.get("RAY_PYTEST_USE_GPU", 0))


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


@ray.remote
class DummyDDPModel(DDPModel):

    def __init__(self, num_layers: int, layer_size: int):
        super().__init__(num_layers)
        self._layer_size = layer_size
        self._weights = [
            torch.ones(
                (layer_size, layer_size),
                dtype=torch.float16,
                device=self._device,
            )
            * 100
            for _ in range(num_layers)
        ]
        self._grad = None
        self._lr = 1

    def start_train(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self._device)

    def get_tensor(self, value: int) -> torch.Tensor:
        shape = (self._layer_size,)
        dtype = torch.float16
        return torch.ones(shape, dtype=dtype, device=self._device) * value

    def forward(self, layer_idx: int, input: torch.Tensor) -> torch.Tensor:
        return self.get_tensor(layer_idx)

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, None]:
        y = y.to(self._device)
        return (pred - y, None)

    # @ray.method(num_returns=2)
    def backward(
        self, layer_idx: int, grad: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bp_grad, _ = grad
        self._grad = self.get_tensor(layer_idx)
        return bp_grad, self._grad

    def update(self, layer_idx: int, grad: torch.Tensor) -> torch.Tensor:
        self._weights[layer_idx] -= grad * self._lr
        return self._weights

    def finish_train(self, *updates: torch.Tensor) -> List[torch.Tensor]:
        return self._weights


@ray.remote
class NNDDPModel(DDPModel):
    def __init__(self, num_layers: int, layer_size: int):
        super().__init__(num_layers)
        self._layer_size = layer_size
        self._weights = [
            torch.ones(
                (layer_size, layer_size),
                dtype=torch.float16,
                device=self._device,
            )
            for _ in range(num_layers)
        ]
        self._biases = [
            torch.ones(
                (layer_size, layer_size),
                dtype=torch.float16,
                device=self._device,
            )
            for _ in range(num_layers)
        ]
        self._activation_fn = torch.relu
        self._lr = 0.0001

        self._pre_activations = None
        self._inputs = None
        self._grads = None

    def start_train(self, x: torch.Tensor) -> torch.Tensor:
        self._pre_activations = []
        self._inputs = []
        self._grads = []
        return x.to(self._device)

    def forward(self, layer_idx: int, input: torch.Tensor) -> torch.Tensor:
        W = self._weights[layer_idx]
        x = input
        self._inputs.append(x)
        b = self._biases[layer_idx]
        y = W @ x + b
        self._pre_activations.append(y)
        z = self._activation_fn(y)

        return z

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, None]:
        y = y.to(self._device)
        grad = 2 * (pred - y) / self._layer_size
        return (grad, None)

    def backward(
        self, layer_idx: int, grad: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def relu_grad(pre_activation: torch.Tensor) -> torch.Tensor:
            return (pre_activation > 0).to(torch.float16)

        bp_grad, _ = grad
        W = self._weights[layer_idx]
        x = self._inputs[layer_idx]
        drelu = relu_grad(self._pre_activations[layer_idx])
        dz = bp_grad * drelu
        dW = dz @ x.T
        db = dz
        dx = W.T @ dz

        dW_flat = dW.view(-1)
        db_flat = db.view(-1)
        update_flat = torch.cat((dW_flat, db_flat))

        return dx, update_flat

    def update(self, layer_idx: int, grad: torch.Tensor) -> torch.Tensor:
        W_size = self._layer_size * self._layer_size
        dW = grad[:W_size].view(self._layer_size, self._layer_size)
        db = grad[W_size:].view(self._layer_size, self._layer_size)
        self._weights[layer_idx] -= dW * self._lr
        self._biases[layer_idx] -= db * self._lr
        return torch.cat(
            (self._weights[layer_idx].view(-1), self._biases[layer_idx].view(-1))
        )

    def finish_train(self, *updates: torch.Tensor) -> List[torch.Tensor]:
        self._pre_activations = None
        self._inputs = None
        self._grads = None
        return updates


class Model(torch.nn.Module):

    def __init__(self, layer_size: int, num_layers: int, device, dtype):
        super(Model, self).__init__()

        self.layers = []
        for _ in range(num_layers):
            self.layers.append(
                torch.nn.Linear(layer_size, layer_size, device=device, dtype=dtype)
            )
            self.layers.append(torch.nn.ReLU())
        self.inputs = []
        self.activations = []
        self._lr = 1e-3
        # self._optimizer = optim.SGD(self._model.parameters(), lr=self._lr)
        self.it = 0

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    # def get_layer(self, layer_idx: int) -> List[torch.nn.Module]:
    #     return [self.layers[2 * layer_idx], self.layers[2 * layer_idx + 1]]

    def forward_layer(self, x, layer_idx):
        self.inputs.append(x)
        linear_layer = self.layers[2 * layer_idx]
        print(f"W: {linear_layer.weight}")
        print(f"b: {linear_layer.bias}")
        print(f"x: {x}")
        y = linear_layer(x)
        relu_activation = self.layers[2 * layer_idx + 1]
        z = relu_activation(y)
        self.activations.append(z)
        print(f"fw x grad fn: {x.grad_fn}")
        print(f"fw y grad fn: {y.grad_fn}")
        print(f"fw z grad fn: {z.grad_fn}")
        print(f"fw W grad fn: {self.layers[2 * layer_idx].weight.grad_fn}")
        return z

    def backward_layer(self, grad: torch.Tensor, layer_idx):
        x = self.inputs[layer_idx]
        # self.it += 1
        # print(self.it)
        print(f"grad: {grad}")
        print(f"x: {x}")
        # x = Variable(x, require_grad=True)
        W = self.layers[2 * layer_idx].weight
        print(f"W: {W}")
        # print(W.requires_grad)
        # print(x.requires_grad)
        print(f"bw W grad fn: {W.grad_fn}")
        print(f"bw x grad fn: {x.grad_fn}")
        print(f"bw grad grad fn: {grad.grad_fn}")
        grad.backward()
        # grad.backward(gradient=grad, inputs=[x, W])
        print(f"x.grad: {x.grad}")
        print(f"W.grad: {W.grad}")
        return x.grad, W.grad

    def update_layer(self, grad, layer_idx):
        with torch.no_grad():
            self.layers[2 * layer_idx].weight -= self._lr * grad
        return self.layers[2 * layer_idx].weight


@ray.remote
class TorchDDPModel(DDPModel):
    def __init__(self, num_layers: int, layer_size: int):
        super().__init__(num_layers)

        self._device = torch_utils.get_devices()[0]
        self._model = Model(layer_size, num_layers)
        self._model = self._model.to(self._device)
        self._criterion = nn.MSELoss()

    def start_train(self, x: torch.Tensor) -> torch.Tensor:
        self._model.zero_grad()
        return x.to(self._device)

    def forward(self, layer_idx: int, input: torch.Tensor) -> torch.Tensor:
        input = input.to(self._device)
        return self._model.forward_layer(input, layer_idx)

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, None]:
        y = y.to(self._device)
        pred = pred.to(self._device)
        print(f"pred grad fn: {pred.grad_fn}")
        loss = self._criterion(pred, y)
        print(f"loss grad fn: {loss.grad_fn}")
        return loss, None

    def backward(
        self, layer_idx: int, grad: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bp_grad, _ = grad
        print(f"layer {layer_idx} grad fn: {bp_grad.grad_fn}")
        bp_grad = bp_grad.to(self._device)
        print(f"layer {layer_idx} after to: {bp_grad.grad_fn}")
        return self._model.backward_layer(bp_grad, layer_idx)

    def update(self, layer_idx: int, grad: torch.Tensor) -> torch.Tensor:
        grad = grad.to(self._device)
        return self._model.update_layer(grad, layer_idx)

    def finish_train(self, *updates: torch.Tensor) -> List[torch.Tensor]:
        return updates


def run_experiment(model: Type[DDPModel]) -> None:
    actor_cls = model.options(num_gpus=1)
    num_layers = 2
    layer_size = 10
    num_actors = 2
    actors = [actor_cls.remote(num_layers, layer_size) for _ in range(num_actors)]

    shape = (num_actors * layer_size, layer_size)
    dtype = torch.float16
    X = torch.ones(shape, dtype=dtype, requires_grad=True) * 10
    Y = torch.ones(shape, dtype=dtype) * 100

    xs = torch.tensor_split(X, num_actors)
    ys = torch.tensor_split(Y, num_actors)
    # print(xs)
    # print(ys)

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
        ends = [actor.finish_train.bind(*output[i]) for i, actor in enumerate(actors)]
        dag = MultiOutputNode(ends)

    compiled_dag = dag.experimental_compile()
    ref = compiled_dag.execute(*xs, *ys)
    result = ray.get(ref)
    print(result)

    compiled_dag.teardown()

    for actor in actors:
        ray.kill(actor)


def main() -> None:
    ray.init()
    if not USE_GPU:
        return
    if sum(node["Resources"].get("GPU", 0) for node in ray.nodes()) < 2:
        return

    # run_experiment(DummyDDPModel)
    # run_experiment(NNDDPModel)
    run_experiment(TorchDDPModel)


if __name__ == "__main__":
    main()
