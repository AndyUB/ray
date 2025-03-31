import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.profiler import ProfilerActivity, profile
from torch.nn.utils import parameters_to_vector

import ray
from ..common import millis_to_micros
from .model import BucketParameter, Shard, shard_model, LinearModel


@ray.remote
class LinearActor:
    def __init__(
        self,
        layer_size: int,
        num_layers: int,
        num_units: int,
        num_actors: int,
        device: torch.device,
        tracing: bool,
        profile_path: Optional[str] = None,
    ):
        self.seed = 998244353

        self.layer_size = layer_size
        self.num_layers = num_layers
        self.num_units = num_units
        self.num_actors = num_actors
        self.device = device
        self.tracing = tracing

        self.shards: List[Shard] = []
        self.input: Optional[torch.Tensor] = None
        self.target: Optional[torch.Tensor] = None
        self.intermediates: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.criterion = torch.nn.MSELoss()

        self.it = 0
        self.events: Dict[str, Any] = {}
        self.elapses: Dict[str, List] = defaultdict(list)

        self.profile_path = profile_path
        self.profile = None
        if self.profile_path:
            self.init_profiling()

    def init_profiling(self) -> None:
        self.profile = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        )
        self.profile.__enter__()

    def finish_profiling(self) -> None:
        self.profile.__exit__(None, None, None)
        self.profile.export_chrome_trace(self.profile_path)

    def init_and_shard_model(self) -> List[List[Shard]]:
        bparams = [
            BucketParameter(
                layer_size=self.layer_size,
                num_layers=self.num_layers // self.num_units,
                device=self.device,
            )
            for _ in range(self.num_units)
        ]
        torch.manual_seed(2025)
        for bparam in bparams:
            bparam.init_weights()
        actor_to_shards = [[] for _ in range(self.num_actors)]
        for bparam in bparams:
            shards = shard_model(bparam, self.num_actors)
            for rank, shard in enumerate(shards):
                actor_to_shards[rank].append(shard)
        return actor_to_shards

    def set_shards(self, shards: List[Shard]) -> None:
        self.shards = [shard.to(self.device) for shard in shards]

    def init_training(self) -> None:
        torch.manual_seed(self.seed)
        self.seed += 1
        self.input = torch.randn(
            (1, self.layer_size),
            device=self.device,
            requires_grad=True,
        )
        self.target = torch.randn(
            (1, self.layer_size),
            device=self.device,
        )
        self.intermediates = []
        torch.cuda.synchronize()

    def init_tracing(self) -> None:
        self.events: Dict[str, torch.cuda.Event] = {
            "start": [],
            "end": [],
            "fw.starts": [],
            "fw.ends": [],
            "comp.loss.starts": [],
            "comp.loss.ends": [],
            "bw.loss.starts": [],
            "bw.loss.ends": [],
            "bw.grad.pre.starts": [],
            "bw.grad.pre.ends": [],
            "bw.grad.intra.starts": [],
            "bw.grad.intra.ends": [],
            "bw.grad.post.starts": [],
            "bw.grad.post.ends": [],
            "bw.upd.starts": [],
            "bw.upd.ends": [],
        }

    def update_tracing(self, key: str) -> None:
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        assert key in self.events
        self.events[key].append(event)

    def finish_tracing(self) -> None:
        torch.cuda.synchronize()
        logger = logging.getLogger(__name__)
        logger.warning(f"Actor finished iteration {self.it}")
        self.it += 1
        if self.it <= 1:
            return

        assert len(self.events["start"]) == 1
        assert len(self.events["end"]) == 1
        total = self.events["start"][0].elapsed_time(self.events["end"][0])

        def log(key: str, total_ms: float, count: int = 1) -> None:
            total_us = millis_to_micros(total_ms)
            self.elapses[key].append(total_us)
            if count == 1:
                logger.warning(
                    f"{key}: {total_us} us, percent: {round(total_ms / total * 100, 1)}%"
                )
            else:
                avg_us = round(total_us / count)
                logger.warning(
                    f"{key}: {total_us} us, avg: {avg_us} us, count: {count}, percent: {round(total_ms / total * 100, 1)}%"
                )

        log(
            "actor.total",
            total,
        )
        if self.tracing:
            log(
                "fw.total",
                self.events["fw.starts"][0].elapsed_time(self.events["fw.ends"][-1]),
                len(self.events["fw.starts"]),
            )
            assert len(self.events["comp.loss.starts"]) == 1
            assert len(self.events["comp.loss.ends"]) == 1
            log(
                "loss.total",
                self.events["comp.loss.starts"][0].elapsed_time(
                    self.events["comp.loss.ends"][0]
                ),
            )
            bw_total = self.events["bw.loss.starts"][0].elapsed_time(
                self.events["bw.upd.ends"][-1]
            )
            assert len(self.events["bw.loss.starts"]) == 1
            assert len(self.events["bw.loss.ends"]) == 1
            bw_loss = self.events["bw.loss.starts"][0].elapsed_time(
                self.events["bw.loss.ends"][0]
            )
            bw_grad_pre = sum(
                [
                    bw_grad_pre_start.elapsed_time(bw_grad_pre_end)
                    for bw_grad_pre_start, bw_grad_pre_end in zip(
                        self.events["bw.grad.pre.starts"],
                        self.events["bw.grad.pre.ends"],
                    )
                ]
            )
            bw_grad_intra = sum(
                [
                    bw_grad_intra_start.elapsed_time(bw_grad_intra_end)
                    for bw_grad_intra_start, bw_grad_intra_end in zip(
                        self.events["bw.grad.intra.starts"],
                        self.events["bw.grad.intra.ends"],
                    )
                ]
            )
            bw_grad_post = sum(
                [
                    bw_grad_post_start.elapsed_time(bw_grad_post_end)
                    for bw_grad_post_start, bw_grad_post_end in zip(
                        self.events["bw.grad.post.starts"],
                        self.events["bw.grad.post.ends"],
                    )
                ]
            )
            bw_upd = sum(
                [
                    bw_upd_start.elapsed_time(bw_upd_end)
                    for bw_upd_start, bw_upd_end in zip(
                        self.events["bw.upd.starts"], self.events["bw.upd.ends"]
                    )
                ]
            )
            bw_grad_wo_loss_upd = bw_total - bw_loss - bw_upd
            log("bw.total", bw_total)
            log("bw.loss", bw_loss)
            log("bw.grad.pre", bw_grad_pre, len(self.events["bw.grad.pre.starts"]))
            log(
                "bw.grad.intra", bw_grad_intra, len(self.events["bw.grad.intra.starts"])
            )
            log("bw.grad.post", bw_grad_post, len(self.events["bw.grad.post.starts"]))
            log("bw.grad.wo.loss_upd", bw_grad_wo_loss_upd)
            log("bw.upd", bw_upd, len(self.events["bw.upd.starts"]))
        logger.warning("")

    def fetch_weights(self) -> List[torch.Tensor]:
        weights = [shard.sharded_param for shard in self.shards]
        return weights

    def fetch_traces(self) -> Dict[str, List[float]]:
        return self.elapses

    def get_input(self, _) -> torch.Tensor:
        assert self.input is not None
        return self.input

    def get_target(self, _) -> torch.Tensor:
        assert self.target is not None
        return self.target

    def get_shard(self, idx: int, _) -> torch.Tensor:
        assert self.shards
        return self.shards[idx].sharded_param

    def forward(self, idx: int, flat_param: torch.Tensor, input: torch.Tensor) -> None:
        if idx == 0:
            self.update_tracing("start")
        if self.tracing:
            self.update_tracing("fw.starts")

        shard = self.shards[idx]
        shard.set_flat_param(flat_param)
        pred = shard.forward(input)

        if idx < len(self.shards) - 1:
            pred_as_input = pred.detach().requires_grad_(True)
        else:
            pred_as_input = pred
        self.intermediates.append((pred, pred_as_input))

        if idx < len(self.shards) - 1:
            shard.free_peer_shards()
        if self.tracing:
            self.update_tracing("fw.ends")
        return pred_as_input

    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.tracing:
            self.update_tracing("comp.loss.starts")
        loss = self.criterion(pred, target)
        if self.tracing:
            self.update_tracing("comp.loss.ends")
        return loss

    def backward_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.tracing:
            self.update_tracing("bw.loss.starts")
        loss.backward()
        shard = self.shards[-1]
        flat_grad = shard.get_flat_grad()
        shard.free_peer_shards()
        if self.tracing:
            self.update_tracing("bw.loss.ends")
        return flat_grad

    def backward_pre(self, idx: int, flat_param: torch.Tensor) -> None:
        if self.tracing:
            self.update_tracing("bw.grad.pre.starts")
        shard = self.shards[idx]
        shard.set_flat_param(flat_param)
        if self.tracing:
            self.update_tracing("bw.grad.pre.ends")
        return None

    def backward_intra(
        self, idx: int, _flat_param: torch.Tensor, _backward_pre
    ) -> torch.Tensor:
        if self.tracing:
            self.update_tracing("bw.grad.intra.starts")
        # [NOTE] backward_pre.
        # shard = self.shards[idx]
        # shard.set_flat_param(flat_param)
        pred, pred_as_input = self.intermediates[idx]
        grad = pred_as_input.grad
        if self.tracing:
            self.update_tracing("bw.grad.pre.ends")
        pred.backward(grad)
        # [NOTE] backward_post.
        # flat_grad = shard.get_flat_grad()
        # shard.free_peer_shards()
        if self.tracing:
            self.update_tracing("bw.grad.intra.ends")
        # return flat_grad
        return None

    def backward_post(
        self, idx: int, _flat_param: torch.Tensor, _backward_intra
    ) -> torch.Tensor:
        if self.tracing:
            self.update_tracing("bw.grad.post.starts")
        shard = self.shards[idx]
        flat_grad = shard.get_flat_grad()
        shard.free_peer_shards()
        if self.tracing:
            self.update_tracing("bw.grad.post.ends")
        return flat_grad

    def update(self, idx: int, grad: torch.Tensor, grad_passed: bool) -> None:
        if self.tracing:
            self.update_tracing("bw.upd.starts")
        if grad_passed:
            grad /= self.num_actors
        self.shards[idx].update(grad, grad_passed)
        if self.tracing:
            self.update_tracing("bw.upd.ends")
        if idx == 0:
            self.update_tracing("end")

    def save_shard_weights(self, rank: int, model_file: str) -> List[torch.Tensor]:
        sharded_params = {
            idx: shard.sharded_param for idx, shard in enumerate(self.shards)
        }
        return sharded_params
        # torch.save(sharded_params, f"{model_file}_rank{rank}.pt")

    def restore_and_save_model(
        self, shards: List[List[torch.Tensor]], model_file: str
    ) -> None:
        for unit in range(self.num_units):
            flat_param = parameters_to_vector([shard[unit] for shard in shards])
            self.shards[unit].set_flat_param(flat_param)

        model = LinearModel(
            self.layer_size, self.num_layers, self.num_units, self.device
        )
        for unit, bucket in enumerate(model.buckets):
            bucket: BucketParameter
            bucket.set_weights(self.shards[unit].model.fetch_weights())
        torch.save(model.state_dict(), model_file)
