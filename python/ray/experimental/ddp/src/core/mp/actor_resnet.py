import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import torch
from torch.profiler import profile, ProfilerActivity

import ray
import ray._private.worker
from ..common import secs_to_micros
from .resnet import resnet152, resnet152_mp
from ray.experimental.channel import ChannelContext
from ray.experimental.channel.torch_tensor_nccl_channel import _NcclGroup


@ray.remote
class ResnetActor:
    def __init__(
        self,
        rank: int,
        num_models: int,
        num_actors: int,
        device: torch.device,
        check_tracing: bool,
    ):
        self.resnet = resnet152_mp(weights=True)
        self.models = [mod.to(device) for mod in self.resnet.bucket_modules]
        assert num_models == len(self.models)

        self.rank = rank
        self.num_models = num_models
        self.num_actors = num_actors
        self.device = device
        self.check_tracing = check_tracing

        logger = logging.getLogger(__name__)
        for model in self.models:
            size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            logger.warning(f"Model size: {size_bytes / 1024 / 1024} MB")
        self.intermediates: List[torch.Tensor, torch.Tensor] = []

        self.it = 0
        self.time: Dict[str, Any] = {}
        self.elapses: Dict[str, List] = defaultdict(list)

        self.nccl_group: _NcclGroup = None
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
            record_shapes=True,
        )

    def init_weights(self) -> None:
        raise NotImplementedError

    def start_profile(self) -> None:
        self.profiler.__enter__()

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

    def update_time(self, key: str) -> None:
        timestamp = time.perf_counter()
        if key not in self.time:
            self.time[key] = timestamp
        else:
            assert isinstance(self.time[key], list)
            self.time[key].append(timestamp)

    def init_tracing(self) -> None:
        self.time: Dict[str, Any] = {
            "forward_starts": [],
            "forward_ends": [],
            "backward_starts": [],
            "backward_ends": [],
            # "allreduce_starts": [],
            # "allreduce_ends": [],
            "update_starts": [],
            "update_ends": [],
        }
        ray._private.worker.global_worker.get_serialization_context().set_tracing(
            enable_tracing=True, clear_traces=True
        )

    def finish_tracing(self) -> None:
        ray._private.worker.global_worker.get_serialization_context().set_tracing(
            enable_tracing=False
        )
        traces = (
            ray._private.worker.global_worker.get_serialization_context().get_traces()
        )

        logger = logging.getLogger(__name__)
        logger.warning(f"Actor {self.rank} finished iteration {self.it}")
        self.it += 1
        # if self.it <= 1:
        #     return

        total = self.time["end"] - self.time["start"]

        def log(key: str, elapse: float):
            self.elapses[key].append(secs_to_micros(elapse))
            logger.warning(
                f"{key} elapse: {secs_to_micros(elapse)} us, percent: {round(elapse / total * 100, 1)}%"
            )

        log(
            "actor.total",
            total,
        )
        if self.check_tracing:
            log(
                "fw.total",
                self.time["forward_ends"][-1] - self.time["forward_starts"][0],
            )
            bw_total = self.time["update_ends"][-1] - self.time["backward_starts"][0]
            bw_backward = sum(
                [
                    self.time["backward_ends"][i] - self.time["backward_starts"][i]
                    for i in range(self.num_models)
                ]
            )
            bw_update = sum(
                [
                    self.time["update_ends"][i] - self.time["update_starts"][i]
                    for i in range(self.num_models)
                ]
            )
            # bw_allreduce = sum(
            #     [
            #         self.time["allreduce_ends"][i] - self.time["allreduce_starts"][i]
            #         for i in range(self.num_models)
            #     ]
            # )
            # bw_others = bw_total - bw_backward - bw_update - bw_allreduce
            bw_others = bw_total - bw_backward - bw_update
            log("bw.total", bw_total)
            log("bw.backward", bw_backward)
            # log("bw.allreduce", bw_allreduce)
            log("bw.others", bw_others)
            log("bw.update", bw_update)
            # logger.warning("")
            # for i in range(len(self.time["backward_starts"])):
            #     log(
            #         f"bw.backward.{i}",
            #         self.time["backward_ends"][i] - self.time["backward_starts"][i],
            #     )

            sum_ser = sum((end - start) for (start, end) in traces["serialization"])
            log("serialization", sum_ser)
            sum_deser = sum((end - start) for (start, end) in traces["deserialization"])
            log("deserialization", sum_deser)
        logger.warning("")

    def fetch_weights(self) -> List[torch.Tensor]:
        raise NotImplementedError

    def fetch_traces(self) -> Dict[str, List[float]]:
        return self.elapses

    def fetch_profile(self) -> None:
        self.profiler.__exit__(None, None, None)
        logger = logging.getLogger(__name__)
        logger.warning(
            self.profiler.key_averages().table(
                sort_by="self_cpu_time_total", row_limit=20
            )
        )
        logger.warning(
            self.profiler.key_averages().table(
                sort_by="cpu_time_total",
                row_limit=20,
            )
        )
        self.profiler.export_chrome_trace(f"profile_{self.rank}.json")

    def forward(self, _) -> None:
        self.update_time("start")
        if self.check_tracing:
            self.update_time("forward_starts")
        self.intermediates = []
        input = self.models[0].x
        for i, model in enumerate(self.models):
            pred = model.forward(input)
            if i < len(self.models) - 1:
                input = pred.detach().requires_grad_(True)
            else:
                input = pred
            self.intermediates.append((pred, input))
        if self.check_tracing:
            self.update_time("forward_ends")

    def backward(self, _, idx: int) -> torch.Tensor:
        if self.check_tracing:
            self.update_time("backward_starts")
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
        grads = self.models[idx].backward(
            loss=loss,
            pred=pred,
            grad=grad,
        )
        if self.check_tracing:
            self.update_time("backward_ends")
        return grads

    def set_nccl_group(self, nccl_group_id: int) -> None:
        ctx = ChannelContext.get_current()
        self.nccl_group = ctx.nccl_groups[nccl_group_id]

    def allreduce(self, grad: torch.Tensor) -> torch.Tensor:
        if self.check_tracing:
            self.update_time("allreduce_starts")
        self.nccl_group.allreduce(grad, grad)
        if self.check_tracing:
            self.update_time("allreduce_ends")
            start = self.time["allreduce_starts"][-1]
            end = self.time["allreduce_ends"][-1]
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Actor {self.rank} iteration {self.it} "
                f"allreduce elapse: {secs_to_micros(end - start)} us"
            )
        return grad

    def update(self, grads_cat: torch.Tensor, grads_passed: bool, idx: int) -> None:
        if self.check_tracing:
            self.update_time("update_starts")
        if grads_passed:
            grads_cat /= self.num_actors
        self.models[idx].update(grads_cat, grads_passed)
        if self.check_tracing:
            self.update_time("update_ends")
        if idx == 0:
            self.update_time("end")
