import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from ....core.common import get_timing_event_torch, log_elapses_to_csv, millis_to_micros
from ....core.config import parse_args
from ....core.resnet.model import resnet152_mp

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)
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


def spawn_torch_fsdp(
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

        device = f"cuda:{rank}"
        model = resnet152_mp(weights=True).to(device)
        size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        logger.warning(f"Model size: {size_bytes / 1024 / 1024} MB")

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        torch.manual_seed(998244353)
        fsdp_model = DDP(model, device_ids=[rank])

        elapses = defaultdict(list)

        BATCH_SIZE = 32

        for iter in range(args["num_iters"]):
            x = torch.randn(
                BATCH_SIZE,
                3,
                224,
                224,
                requires_grad=True,
            ).to(
                device,
            )
            y = torch.randint(
                0,
                1000,
                (BATCH_SIZE,),
            ).to(
                device,
            )

            if rank == 0:
                logger.info(f"iter: {iter}")
                logger.info(f"input: {x}")
                logger.info(f"target: {y}")

            torch.cuda.synchronize()
            dist.barrier()
            start = get_timing_event_torch()

            forward_start = get_timing_event_torch()
            pred = fsdp_model(x)
            forward_end = get_timing_event_torch()

            loss_compute_start = get_timing_event_torch()
            loss = criterion(pred, y)
            loss_compute_end = get_timing_event_torch()

            backward_start = get_timing_event_torch()
            loss.backward()
            backward_end = get_timing_event_torch()

            update_start = get_timing_event_torch()
            optimizer.step()
            optimizer.zero_grad()
            update_end = get_timing_event_torch()

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

            log("total", total_ms)
            log("fw.total", forward_start.elapsed_time(forward_end))
            log("loss.compute", loss_compute_start.elapsed_time(loss_compute_end))
            log("bw.bw_ar", backward_start.elapsed_time(backward_end))
            log("bw.update", update_start.elapsed_time(update_end))
            log("barrier", barrier_start.elapsed_time(end))
    finally:
        dist.destroy_process_group()

    ranks_to_elapses[rank] = elapses


if __name__ == "__main__":
    args = parse_args()
    run_torch_fsdp(args)
