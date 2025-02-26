from typing import List, Optional, Tuple

import torch
from torch.nn.utils import parameters_to_vector


class BucketParameter(torch.nn.Module):
    def __init__(
        self,
        layer_size: int,
        num_layers: int,
        device: torch.device,
    ):
        super().__init__()

        self.layer_size = layer_size
        self.num_layers = num_layers
        self.device = device
        self.linear_layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    layer_size,
                    layer_size,
                    bias=False,
                )
                for _ in range(num_layers)
            ]
        )
        self.relu_layers = torch.nn.ModuleList(
            [torch.nn.ReLU() for _ in range(num_layers)]
        )

        self.x = None
        self.y = None
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.linear_layers.parameters(),
            lr=1e-3,
        )

    def init_weights(self) -> None:
        with torch.no_grad():
            for layer in self.linear_layers:
                torch.nn.init.kaiming_uniform_(
                    layer.weight, mode="fan_in", nonlinearity="relu"
                )

    def fetch_weights(self) -> List[torch.Tensor]:
        return [layer.weight.detach() for layer in self.linear_layers]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, relu in zip(self.linear_layers, self.relu_layers):
            y = linear(x)
            z = relu(y)
            x = z
        return z

    def backward(
        self,
        loss: Optional[torch.Tensor] = None,
        pred: Optional[torch.Tensor] = None,
        grad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if loss is not None:
            assert pred is None
            loss.backward()
        elif pred is not None:
            assert grad is not None
            pred.backward(grad)

        grads_cat = parameters_to_vector(
            [layer.weight.grad for layer in self.linear_layers]
        )
        return grads_cat

    def update(self, grads_cat: torch.Tensor, grads_passed: bool) -> None:
        if grads_passed:
            offset = 0
            for layer in self.linear_layers:
                size = layer.weight.numel()
                grad = grads_cat[offset : offset + size].reshape(layer.weight.shape)
                layer.weight.grad = grad
                offset += size
            del grads_cat

        self.optimizer.step()
        self.optimizer.zero_grad()


class Shard(torch.nn.Module):
    def __init__(
        self,
        module: torch.nn.Module,
        shard: torch.Tensor,
        param_metadata: List[Tuple[torch.Size, int]],
    ) -> None:
        super().__init__()

        self.shard = torch.nn.Parameter(shard, requires_grad=True)
        self.shard_size = len(shard)
        self.sharded_module = module
        self.param_metadata = param_metadata
        self.optimizer = torch.optim.Adam([self.shard], lr=1e-3)

    def unwrap(self) -> torch.nn.Parameter:
        return self.shard

    def unshard(self, flat_param: torch.Tensor) -> None:
        unshard_model(self.sharded_module, flat_param, self.param_metadata)

    def free_peer_shards(self) -> None:
        _free_model(self.sharded_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sharded_module(x)

    def flat_grad(self) -> torch.Tensor:
        flat_grad = parameters_to_vector(
            [param.grad for param in self.sharded_module.parameters()]
        )
        return flat_grad

    def update(self, reduced_grad: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        self.shard.grad = reduced_grad
        self.optimizer.step()


def shard_model(model: torch.nn.Module, sharding_factor: int) -> List[Shard]:
    flat_param = parameters_to_vector(model.parameters())
    padding = (sharding_factor - flat_param.numel() % sharding_factor) % sharding_factor
    if padding != 0:
        dtype = flat_param.dtype
        device = flat_param.device
        flat_param = torch.cat(
            [flat_param, torch.zeros((padding,), dtype=dtype, device=device)]
        )
    shard_size = len(flat_param) // sharding_factor
    shards = [
        flat_param[shard_size * i : shard_size * (i + 1)]
        for i in range(sharding_factor)
    ]
    param_metadata = _model_param_metadata(model)
    model = _free_model(model)
    return [Shard(model, shard, param_metadata) for shard in shards]


def _model_param_metadata(model: torch.nn.Module) -> List[Tuple[torch.Size, int]]:
    return [(param.shape, param.numel()) for param in model.parameters()]


def _free_model(model: torch.nn.Module) -> torch.nn.Module:
    def get_first_param():
        for param in model.parameters():
            return param
        raise ValueError("Model has no parameters")

    first_param = get_first_param()
    dtype = first_param.dtype
    device = first_param.device
    empty_tensor = torch.zeros((0,), dtype=dtype, device=device)
    for param in model.parameters():
        param.data = empty_tensor
        param.grad = None
    return model


def unshard_model(
    model: torch.nn.Module,
    flat_param: torch.Tensor,
    param_metedata: List[Tuple[torch.Size, int]],
) -> torch.nn.Module:
    start_idx = 0
    with torch.no_grad():
        for param, metadata in zip(model.parameters(), param_metedata):
            shape, numel = metadata
            end_idx = start_idx + numel
            param.data = flat_param[start_idx:end_idx].reshape(shape)
            start_idx = end_idx
    return model
