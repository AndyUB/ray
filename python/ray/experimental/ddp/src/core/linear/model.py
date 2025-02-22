from typing import List, Optional

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
