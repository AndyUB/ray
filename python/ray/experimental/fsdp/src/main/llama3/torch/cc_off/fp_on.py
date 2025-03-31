import functools
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import BackwardPrefetch
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from .....core.common import (
    get_timing_event_torch,
    log_elapses_to_csv,
    millis_to_micros,
)
from .....core.config import parse_args
from .....core.llama3.model import LLAMA_DEBUG as LLAMA
from .....core.llama3.model import BucketParameterBase, TransformerWrapped

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def run_torch_fsdp(
    args: Dict[str, Any],
) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= args["num_actors"]
    world_size = args["num_actors"]

    mp.set_start_method("spawn", force=True)

    with mp.Manager() as manager:
        ranks_to_elapses = manager.dict()

        mp.spawn(
            spawn_torch_fsdp,
            args=(world_size, ranks_to_elapses, args),
            nprocs=world_size,
            join=True,
        )

        ranks_to_elapses_list = list(ranks_to_elapses[i] for i in range(world_size))

    output_path = args["output_path"]
    latency_prefix = args["latency_prefix"]
    metrics = [
        "total",
        "actor.total",
        "fw.total",
        "bw.total",
        "bw.loss.comp",
        "bw.grad",
        "others.upd",
        "barrier",
    ]
    log_elapses_to_csv(
        ranks_to_elapses_list,
        output_path,
        latency_prefix,
        metrics,
    )


def spawn_torch_fsdp(
    rank: int,
    world_size: int,
    ranks_to_elapses: Dict[int, int],
    args: Dict[str, Any],
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    os.environ["FSDP_USE_FAKE_ALL_GATHER"] = "1"
    os.environ["FSDP_USE_FAKE_REDUCE"] = "1"

    try:
        logger = logging.getLogger(__name__)

        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
        torch.manual_seed(998244353)

        model_args = LLAMA
        logger.info(f"model_args: {model_args}")
        model = TransformerWrapped(model_args).to("cuda")
        size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        logger.warning(f"Model size: {size_bytes / 1024 / 1024} MiB")

        fsdp_model = FSDP(
            model,
            auto_wrap_policy=functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda p: isinstance(p, BucketParameterBase),
            ),
            device_id=device,
        )
        if rank == 0:
            logger.info(f"FSDP model: {fsdp_model}")

        batch_size = args["batch_size"]
        seq_len = args["seq_len"]
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-6)
        elapses = defaultdict(list)

        for iter in range(args["num_iters"]):
            input_ids = torch.randint(
                0,
                model_args.vocab_size,
                (batch_size, seq_len),
                device=device,
            )
            target_ids = torch.randn(
                batch_size,
                seq_len,
                model_args.vocab_size,
                requires_grad=True,
                device=device,
            )

            if rank == 0:
                logger.info(f"iter: {iter}")

            torch.cuda.synchronize()
            dist.barrier()
            start = get_timing_event_torch()

            fw_start = get_timing_event_torch()
            pred = fsdp_model(input_ids)
            fw_end = get_timing_event_torch()

            bw_loss_comp_start = get_timing_event_torch()
            loss = criterion(pred, target_ids)
            bw_loss_comp_end = get_timing_event_torch()

            bw_grad_start = get_timing_event_torch()
            loss.backward()
            bw_grad_end = get_timing_event_torch()

            others_upd_start = get_timing_event_torch()
            optimizer.step()
            optimizer.zero_grad()
            others_upd_end = get_timing_event_torch()

            torch.cuda.synchronize()
            barrier_start = get_timing_event_torch()
            dist.barrier()
            end = get_timing_event_torch()
            torch.cuda.synchronize()

            total_ms = start.elapsed_time(end)

            def log(key: str, elapse_ms: float):
                elapse_us = millis_to_micros(elapse_ms)
                elapses[key].append(elapse_us)
                logger.warning(
                    f"rank: {rank}, {key} elapse: {elapse_us} us, percent: {round(elapse_ms / total_ms * 100, 1)}%"
                )

            if iter > 0:
                log("total", total_ms)
                log("actor.total", total_ms)
                log("fw.total", fw_start.elapsed_time(fw_end))
                log("bw.total", bw_loss_comp_start.elapsed_time(bw_grad_end))
                log("bw.loss.comp", bw_loss_comp_start.elapsed_time(bw_loss_comp_end))
                log("bw.grad", bw_grad_start.elapsed_time(bw_grad_end))
                log("others.upd", others_upd_start.elapsed_time(others_upd_end))
                log("barrier", barrier_start.elapsed_time(end))
                logger.warning("")
    finally:
        dist.destroy_process_group()

    ranks_to_elapses[rank] = elapses


if __name__ == "__main__":
    args = parse_args()
    run_torch_fsdp(args)
