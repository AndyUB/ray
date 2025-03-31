import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from ....core.common import (
    get_end_time,
    get_start_time,
    log_elapses_to_csv,
    ms_to_micros,
)
from ....core.config import parse_args
from ....core.llama3.model import LLAMA_1B, TransformerBP

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def run_torch_ddp(
    args: Dict[str, Any],
) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= args["num_actors"]
    world_size = args["num_actors"]

    mp.set_start_method("spawn", force=True)

    with mp.Manager() as manager:
        ranks_to_elapses = manager.dict()

        mp.spawn(
            spawn_torch_ddp,
            args=(world_size, ranks_to_elapses, args),
            nprocs=world_size,
            join=True,
        )

        ranks_to_elapses_list = list(ranks_to_elapses[i] for i in range(world_size))

    output_path = args["output_path"]
    latency_prefix = args["latency_prefix"]
    metrics = [
        "total",
        "fw.total",
        "loss.compute",
        "bw.bw_ar",
        "bw.update",
        "barrier",
    ]
    log_elapses_to_csv(
        ranks_to_elapses_list,
        output_path,
        latency_prefix,
        metrics,
    )


def spawn_torch_ddp(
    rank: int,
    world_size: int,
    ranks_to_elapses: Dict[int, int],
    args: Dict[str, Any],
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    try:
        logger = logging.getLogger(__name__)

        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        torch.cuda.set_device(rank)
        torch.manual_seed(998244353)

        model_args = LLAMA_1B
        logger.info(f"model_args: {model_args}")
        model = TransformerBP(model_args).to("cuda")
        size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        logger.warning(f"Model size: {size_bytes / 1024 / 1024} MiB")

        ddp_model = DDP(model, device_ids=[rank])

        batch_size = 1
        seq_len = 1024
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        elapses = defaultdict(list)

        for iter in range(args["num_iters"]):
            input_ids = torch.randint(
                0,
                model_args.vocab_size,
                (batch_size, seq_len),
            ).to("cuda")
            target_ids = torch.randn(
                batch_size,
                seq_len,
                model_args.vocab_size,
                requires_grad=True,
            ).to("cuda")

            if rank == 0:
                logger.info(f"iter: {iter}")
                logger.info(f"input_ids: {input_ids}")
                logger.info(f"target_ids: {target_ids}")

            torch.cuda.synchronize()
            dist.barrier()
            start = get_start_time()

            forward_start = get_start_time()
            pred = ddp_model(input_ids)
            forward_end = get_end_time()

            loss_compute_start = get_start_time()
            loss = criterion(pred, target_ids)
            loss_compute_end = get_end_time()

            backward_start = get_start_time()
            loss.backward()
            backward_end = get_end_time()

            update_start = get_start_time()
            optimizer.step()
            optimizer.zero_grad()
            update_end = get_end_time()

            barrier_start = get_start_time()
            dist.barrier()
            end = get_end_time()

            total_ms = (end - start) * 1e3

            def log(key: str, elapse_ms: float):
                elapse_us = ms_to_micros(elapse_ms)
                elapses[key].append(elapse_us)
                logger.warning(
                    f"rank: {rank}, {key} elapse: {elapse_us} us, percent: {round(elapse_ms / total_ms * 100, 1)}%"
                )

            log("total", total_ms)
            log("fw.total", (forward_end - forward_start) * 1e3)
            log("loss.compute", (loss_compute_end - loss_compute_start) * 1e3)
            log("bw.bw_ar", (backward_end - backward_start) * 1e3)
            log("bw.update", (update_end - update_start) * 1e3)
            log("barrier", (end - barrier_start) * 1e3)
    finally:
        dist.destroy_process_group()

    ranks_to_elapses[rank] = elapses


if __name__ == "__main__":
    args = parse_args()
    run_torch_ddp(args)
