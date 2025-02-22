import logging
from collections import defaultdict
from typing import Any, Dict, List

import torch

import ray
from ..common import ms_to_micros
from .model import resnet152_mp as resnet_mp

# from .resnet import resnet101_mp as resnet_mp
# from .resnet import resnet50_mp as resnet_mp


@ray.remote
class ResnetActor:
    def __init__(
        self,
        rank: int,
        num_partitions: int,
        num_actors: int,
        device: torch.device,
        tracing: bool,
    ):
        self.resnet = resnet_mp(weights=True)
        self.models = [bparam.to(device) for bparam in self.resnet.bucket_params]
        assert num_partitions == len(self.models)

        self.rank = rank
        self.num_partitions = num_partitions
        self.num_actors = num_actors
        self.device = device
        self.tracing = tracing

        logger = logging.getLogger(__name__)
        for model in self.models:
            size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            logger.warning(f"Model size: {size_bytes / 1024 / 1024:.2f} MiB")
        self.intermediates: List[torch.Tensor, torch.Tensor] = []

        self.it = 0
        self.events: Dict[str, Any] = {}
        self.elapses: Dict[str, List] = defaultdict(list)

    def init_weights(self) -> None:
        raise NotImplementedError

    def init_training(self, batch_size: int) -> None:
        self.models[0].x = torch.randn(
            batch_size,
            3,
            224,
            224,
            requires_grad=True,
        ).to(
            self.device,
        )
        self.models[-1].y = torch.randint(
            0,
            1000,
            (batch_size,),
        ).to(
            self.device,
        )

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
        logger.warning(f"Actor {self.rank} finished iteration {self.it}")
        self.intermediates = []
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
                    for i in range(self.num_partitions)
                ]
            )
            bw_update = sum(
                [
                    self.events["update_starts"][i].elapsed_time(
                        self.events["update_ends"][i]
                    )
                    for i in range(self.num_partitions)
                ]
            )
            bw_others = bw_total - bw_backward - bw_update
            log("bw.total", bw_total)
            log("bw.backward", bw_backward)
            log("bw.others", bw_others)
            log("bw.update", bw_update)
        logger.warning("")

    def fetch_weights(self) -> List[torch.Tensor]:
        raise NotImplementedError

    def fetch_traces(self) -> Dict[str, List[float]]:
        return self.elapses

    def forward(self, _) -> None:
        self.update_tracing("start")
        if self.tracing:
            self.update_tracing("forward_starts")
        self.intermediates = []
        input = self.models[0].x
        for i, model in enumerate(self.models):
            pred = model.forward(input)
            if i < len(self.models) - 1:
                input = pred.detach().requires_grad_(True)
            else:
                input = pred
            self.intermediates.append((pred, input))
        if self.tracing:
            self.update_tracing("forward_ends")

    def backward(self, _, idx: int) -> torch.Tensor:
        if self.tracing:
            self.update_tracing("backward_starts")
        if idx == len(self.models) - 1:
            loss = self.models[idx].criterion(
                self.intermediates[idx][0],
                self.models[idx].y,
            )
            pred = None
            grad = None
        else:
            loss = None
            pred, input = self.intermediates[idx]
            grad = input.grad
            self.intermediates[idx] = (None, None)
        grads = self.models[idx].backward(
            loss=loss,
            pred=pred,
            grad=grad,
        )
        if self.tracing:
            self.update_tracing("backward_ends")
        return grads

    def update(self, grads_cat: torch.Tensor, grads_passed: bool, idx: int) -> None:
        if self.tracing:
            self.update_tracing("update_starts")
        if grads_passed:
            grads_cat /= self.num_actors
        self.models[idx].update(grads_cat, grads_passed)
        if self.tracing:
            self.update_tracing("update_ends")
        if idx == 0:
            self.update_tracing("end")
