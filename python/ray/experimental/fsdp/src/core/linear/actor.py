import logging
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch

import ray
from ..common import ms_to_micros
from .model import BucketParameter, shard_model, Shard


@ray.remote
class LinearActor:
    def __init__(
        self,
        layer_size: int,
        num_layers_per_unit: int,
        num_units: int,
        num_actors: int,
        device: torch.device,
        tracing: bool,
    ):
        self.layer_size = layer_size
        self.num_layers_per_unit = num_layers_per_unit
        self.num_units = num_units
        self.num_actors = num_actors
        self.device = device
        self.tracing = tracing

        self.shards: List[Shard] = []
        self.input: torch.Tensor = None
        self.intermediates: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.output: torch.Tensor = None

        self.criterion = torch.nn.MSELoss()

        self.it = 0
        self.events: Dict[str, Any] = {}
        self.elapses: Dict[str, List] = defaultdict(list)

    def init_and_shard_model(self) -> List[List[Shard]]:
        sharding_factor = self.num_actors
        fsdp_units = [
            BucketParameter(
                layer_size=self.layer_size,
                num_layers=self.num_layers_per_unit,
                device=self.device,
            )
            for _ in range(self.num_units)
        ]
        shards_across_units = [[] for _ in range(sharding_factor)]
        for unit in fsdp_units:
            single_unit_shards = shard_model(unit, sharding_factor)
            for rank, shard in enumerate(single_unit_shards):
                shards_across_units[rank].append(shard)
        return shards_across_units

    def set_shards(self, shards: List[Shard]) -> None:
        self.shards = shards
        for shard in self.shards:
            shard.to(self.device)

    def init_training(self) -> None:
        self.input = torch.randn(
            (1, self.layer_size), device=self.device, requires_grad=True
        )
        self.output = torch.randn(
            (1, self.layer_size), device=self.device, requires_grad=False
        )
        self.intermediates = []

    def update_tracing(self, key: str) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        if key not in self.events:
            self.events[key] = event
        else:
            assert isinstance(self.events[key], list)
            self.events[key].append(event)

    def init_tracing(self) -> None:
        self.events: Dict[str, torch.cuda.Event] = {
            "forward_starts": [],
            "forward_ends": [],
            "backward_starts": [],
            "backward_ends": [],
            "update_starts": [],
            "update_ends": [],
        }

    def finish_tracing(self) -> None:
        torch.cuda.synchronize()
        logger = logging.getLogger(__name__)
        logger.warning(f"Actor finished iteration {self.it}")
        self.it += 1
        if self.it <= 1:
            return

        total = self.events["start"].elapsed_time(self.events["end"])

        def log(key: str, elapse_ms: float):
            elapse_us = ms_to_micros(elapse_ms)
            self.elapses[key].append(elapse_us)
            logger.warning(
                f"{key} elapse: {elapse_us} us, percent: {round(elapse_ms / total * 100, 1)}%"
            )

        log(
            "actor.total",
            total,
        )
        if self.tracing:
            log(
                "fw.total",
                self.events["forward_starts"][0].elapsed_time(
                    self.events["forward_ends"][-1]
                ),
            )
            bw_total = self.events["backward_starts"][0].elapsed_time(
                self.events["update_ends"][-1]
            )
            bw_backward = sum(
                [
                    self.events["backward_starts"][i].elapsed_time(
                        self.events["backward_ends"][i]
                    )
                    for i in range(self.num_units)
                ]
            )
            bw_update = sum(
                [
                    self.events["update_starts"][i].elapsed_time(
                        self.events["update_ends"][i]
                    )
                    for i in range(self.num_units)
                ]
            )
            bw_others = bw_total - bw_backward - bw_update
            log("bw.total", bw_total)
            log("bw.backward", bw_backward)
            log("bw.others", bw_others)
            log("bw.update", bw_update)
        logger.warning("")

    def fetch_weights(self) -> List[torch.Tensor]:
        weights = []
        for shard in self.shards:
            weights.extend(shard.unwrap())
        return weights

    def fetch_traces(self) -> Dict[str, List[float]]:
        return self.elapses

    def get_input(self, _) -> torch.Tensor:
        assert self.input is not None
        return self.input

    def get_output(self, _) -> torch.Tensor:
        assert self.output is not None
        return self.output

    def get_shard(self, unit: int, _) -> torch.Tensor:
        assert self.shards
        return self.shards[unit].unwrap()

    def forward(
        self, unit: int, unsharded_param: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        shard = self.shards[unit]
        shard.unshard(unsharded_param)
        if unit == 0:
            self.update_tracing("start")
        if self.tracing:
            self.update_tracing("forward_starts")
        pred: torch.Tensor = shard(x)
        if unit == self.num_units - 1:
            next_layer_input = pred
        else:
            next_layer_input = pred.detach().requires_grad_(True)
        self.intermediates.append((pred, next_layer_input))
        if self.tracing:
            self.update_tracing("forward_ends")
        if unit != self.num_units - 1:
            shard.free_peer_shards()
        return next_layer_input

    def compute_loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.criterion(pred, y)

    def backward_loss(self, loss: torch.Tensor) -> torch.Tensor:
        loss.backward()
        shard = self.shards[self.num_units - 1]
        flat_grad = shard.flat_grad()
        shard.free_peer_shards()
        return flat_grad

    def backward(self, unit: int, unsharded_param: torch.Tensor) -> torch.Tensor:
        shard = self.shards[unit]
        shard.unshard(unsharded_param)
        if self.tracing:
            self.update_tracing("backward_starts")
        pred, next_layer_input = self.intermediates[unit]
        grad = next_layer_input.grad
        pred.backward(gradient=grad)
        if self.tracing:
            self.update_tracing("backward_ends")
        flat_grad = shard.flat_grad()
        shard.free_peer_shards()
        return flat_grad

    def update(self, unit: int, reduced_grad: torch.Tensor) -> None:
        if self.tracing:
            self.update_tracing("update_starts")
        reduced_grad /= self.num_actors
        shard = self.shards[unit]
        shard.update(reduced_grad)
        if self.tracing:
            self.update_tracing("update_ends")
        if unit == 0:
            self.update_tracing("end")
