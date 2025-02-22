import logging
from collections import defaultdict
from typing import Any, Dict, List

import torch

import ray
from ...core.llama3.model import TransformerBP
from ..common import ms_to_micros

logger = logging.getLogger(__name__)


@ray.remote
class LlamaActor:
    def __init__(
        self,
        model_args,
        rank: int,
        num_partitions: int,
        num_actors: int,
        tracing: bool,
    ):
        torch.cuda.set_device(0)
        torch.manual_seed(998244353)

        logger.info(f"model_args: {model_args}")
        self.model_args = model_args
        self.model = TransformerBP(model_args).to("cuda")
        self.bparams = self.model.bparams
        assert len(self.bparams) == num_partitions

        self.rank = rank
        self.num_partitions = num_partitions
        self.num_actors = num_actors
        self.tracing = tracing

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)

        self.it = 0
        self.elapses: Dict[str, List] = defaultdict(list)

    def init_training(self) -> None:
        batch_size = 1
        seq_len = 1024

        self.input_ids = torch.randint(
            0,
            self.model_args.vocab_size,
            (batch_size, seq_len),
        ).to("cuda")
        self.target_ids = torch.randn(
            batch_size,
            seq_len,
            self.model_args.vocab_size,
            requires_grad=True,
        ).to("cuda")

        self.events: Dict[str, List] = {
            "start": [],
            "end": [],
            "forward_starts": [],
            "forward_ends": [],
            "backward_starts": [],
            "backward_ends": [],
            "update_starts": [],
            "update_ends": [],
        }

    def update_tracing(self, key: str) -> None:
        if self.tracing or key in ["start", "end"]:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            assert key in self.events
            self.events[key].append(event)

    def finish_tracing(self) -> None:
        torch.cuda.synchronize()
        logger = logging.getLogger(__name__)
        logger.warning(f"Actor {self.rank} finished iteration {self.it}")
        self.intermediates = []
        self.it += 1
        if self.it <= 1:
            return

        total = self.events["start"][0].elapsed_time(self.events["end"][0])

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

    def fetch_traces(self) -> Dict[str, List[float]]:
        return self.elapses

    def forward(self, _) -> torch.Tensor:
        self.update_tracing("start")
        self.update_tracing("forward_starts")
        self.intermediates = []
        tokens = self.input_ids
        input, freqs_cis, mask = None, None, None
        for i, bp in enumerate(self.bparams):
            if i == 0:
                pred = bp.forward(tokens)
                freqs_cis, mask = bp.post_hook(tokens, pred)
            elif i < len(self.bparams) - 1:
                pred = bp.forward_transformer(input, 0, freqs_cis, mask)
            else:
                pred = bp.forward(bp.pre_hook(input))
            if i < len(self.bparams) - 1:
                input = pred.detach().requires_grad_(True)
            else:
                input = pred
            self.intermediates.append((pred, input))
        self.update_tracing("forward_ends")
        return pred

    def backward(self, _, idx: int) -> torch.Tensor:
        self.update_tracing("backward_starts")
        if idx == len(self.bparams) - 1:
            loss = self.bparams[idx].criterion(
                self.intermediates[idx][0],
                self.target_ids,
            )
            pred = None
            grad = None
        else:
            loss = None
            pred, input = self.intermediates[idx]
            grad = input.grad
        grads = self.bparams[idx].backward(
            loss=loss,
            pred=pred,
            grad=grad,
        )
        self.update_tracing("backward_ends")
        return grads

    def backward_aio(self, _) -> torch.Tensor:
        for i in reversed(range(len(self.bparams))):
            self.backward(_, i)

    def copy(self, grads_cat: torch.Tensor, grads_passed: bool, idx: int) -> None:
        if grads_passed:
            grads_cat /= self.num_actors
        self.bparams[idx].copy(grads_cat, grads_passed)

    def copy_aio(self, _) -> None:
        for i in reversed(range(len(self.bparams))):
            self.copy(_, False, i)

    def step(self, idx: int) -> None:
        self.bparams[idx].step()

    def step_aio(self, _) -> None:
        # [NOTE] It is slower to use a single optimizer.
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        for i in reversed(range(len(self.bparams))):
            self.step(i)

    def update(self, grads_cat: torch.Tensor, grads_passed: bool, idx: int) -> None:
        self.update_tracing("update_starts")
        self.copy(grads_cat, grads_passed, idx)
        self.step(idx)
        self.update_tracing("update_ends")
        if idx == 0:
            self.update_tracing("end")

    def update_aio(self, _) -> None:
        for i in reversed(range(len(self.bparams))):
            self.update(_, False, i)


class _Actor_V5:
    def __init__(self, model_args):
        logger.info(f"model_args: {model_args}")
        self.model_args = model_args
        self.model = TransformerBP(model_args).to("cuda")
        self.bparams = self.model.bparams
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)

    def init_training(self) -> None:
        batch_size = 1
        seq_len = 1024
        self.input_ids = torch.randint(
            0,
            self.model_args.vocab_size,
            (batch_size, seq_len),
        ).to("cuda")
        self.target_ids = torch.randn(
            batch_size,
            seq_len,
            self.model_args.vocab_size,
            requires_grad=True,
        ).to("cuda")

    def forward(self, _) -> torch.Tensor:
        self.intermediates = []
        tokens = self.input_ids
        input, freqs_cis, mask = None, None, None
        for i, bp in enumerate(self.bparams):
            if i == 0:
                pred = bp.forward(tokens)
                freqs_cis, mask = bp.post_hook(tokens, pred)
            elif i < len(self.bparams) - 1:
                pred = bp.forward_transformer(input, 0, freqs_cis, mask)
            else:
                pred = bp.forward(bp.pre_hook(input))
            input = pred
            self.intermediates.append((pred, input))
        return pred

    def backward(self, _) -> None:
        loss = self.criterion(
            self.intermediates[-1][0],
            self.target_ids,
        )
        loss.backward()

    def update(self, _) -> None:
        self.optimizer.step()
        self.optimizer.zero_grad()


class _Actor_V4:
    def __init__(self, model_args):
        logger.info(f"model_args: {model_args}")
        self.model_args = model_args
        self.model = TransformerBP(model_args).to("cuda")
        self.bparams = self.model.bparams
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)

    def init_training(self) -> None:
        batch_size = 1
        seq_len = 1024
        self.input_ids = torch.randint(
            0,
            self.model_args.vocab_size,
            (batch_size, seq_len),
        ).to("cuda")
        self.target_ids = torch.randn(
            batch_size,
            seq_len,
            self.model_args.vocab_size,
            requires_grad=True,
        ).to("cuda")

    def forward(self, _) -> torch.Tensor:
        self.intermediates = []
        tokens = self.input_ids
        input, freqs_cis, mask = None, None, None
        for i, bp in enumerate(self.bparams):
            if i == 0:
                pred = bp.forward(tokens)
                freqs_cis, mask = bp.post_hook(tokens, pred)
            elif i < len(self.bparams) - 1:
                pred = bp.forward_transformer(input, 0, freqs_cis, mask)
            else:
                pred = bp.forward(bp.pre_hook(input))
            if i < len(self.bparams) - 1:
                input = pred.detach().requires_grad_(True)
            else:
                input = pred
            self.intermediates.append((pred, input))
        return pred

    def backward(self, _, idx: int) -> torch.Tensor:
        if idx == len(self.bparams) - 1:
            loss = self.bparams[idx].criterion(
                self.intermediates[idx][0],
                self.target_ids,
            )
            pred = None
            grad = None
        else:
            loss = None
            pred, input = self.intermediates[idx]
            grad = input.grad
        grads = self.bparams[idx].backward(
            loss=loss,
            pred=pred,
            grad=grad,
        )
        return grads

    def backward_aio(self, _) -> torch.Tensor:
        for i in reversed(range(len(self.bparams))):
            self.backward(_, i)

    def _copy(self, grads_cat: torch.Tensor, grads_passed: bool, idx: int) -> None:
        if grads_passed:
            grads_cat /= self.num_actors
        self.bparams[idx].copy(grads_cat, grads_passed)

    def copy_aio(self, _) -> None:
        for i in reversed(range(len(self.bparams))):
            self._copy(_, False, i)

    def _step(self, idx: int) -> None:
        self.bparams[idx].step()

    def step_aio(self, _) -> None:
        # [NOTE] It is slower to use a single optimizer.
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        for i in reversed(range(len(self.bparams))):
            self._step(i)

    def update(self, grads_cat: torch.Tensor, grads_passed: bool, idx: int) -> None:
        self._copy(grads_cat, grads_passed, idx)
        self._step(idx)

    def update_aio(self, _) -> None:
        for i in reversed(range(len(self.bparams))):
            self.update(_, False, i)


class _Actor_V2:
    def __init__(self, model_args):
        model_args.n_layers = 1
        logger.info(f"model_args: {model_args}")
        self.model_args = model_args
        self.model = TransformerBP(model_args).to("cuda")
        self.bparams = self.model.bparams
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-6)

    def init_training(self) -> None:
        batch_size = 1
        seq_len = 1024
        self.input_ids = torch.randint(
            0,
            self.model_args.vocab_size,
            (batch_size, seq_len),
        ).to("cuda")
        self.target_ids = torch.randn(
            batch_size,
            seq_len,
            self.model_args.vocab_size,
            requires_grad=True,
        ).to("cuda")

    def forward(self, _) -> torch.Tensor:
        self.intermediates = []
        tokens = self.input_ids
        input, freqs_cis, mask = None, None, None
        for i, bp in enumerate(self.bparams):
            if i == 0:
                pred = bp.forward(tokens)
                freqs_cis, mask = bp.post_hook(tokens, pred)
            elif i < len(self.bparams) - 1:
                pred = bp.forward_transformer(input, 0, freqs_cis, mask)
            else:
                pred = bp.forward(bp.pre_hook(input))
            if i < len(self.bparams) - 1:
                input = pred.detach().requires_grad_(True)
            else:
                input = pred
            self.intermediates.append((pred, input))
        return pred

    def backward(self, _, idx: int) -> torch.Tensor:
        if idx == len(self.bparams) - 1:
            loss = self.bparams[idx].criterion(
                self.intermediates[idx][0],
                self.target_ids,
            )
            pred = None
            grad = None
        else:
            loss = None
            pred, input = self.intermediates[idx]
            grad = input.grad
        grads = self.bparams[idx].backward(
            loss=loss,
            pred=pred,
            grad=grad,
        )
        return grads

    def backward_aio(self, logits) -> torch.Tensor:
        for i in reversed(range(len(self.bparams))):
            self.backward(None, i)

    def update_aio(self, _) -> None:
        # [VERSION] Exclude optimizer.
        # self.optimizer.step()
        # self.optimizer.zero_grad()
        for bparam in self.bparams:
            bparam.step(None, False)
