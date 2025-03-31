import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from ....core.common import get_timing_event, log_elapses_to_csv, ms_to_micros
from ....core.config import parse_args
from ....core.linear.actor import BucketParameter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)
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

        device = f"cuda:{rank}"
        model = BucketParameter(
            args["layer_size"],
            args["num_layers"],
            device,
        )
        size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        logger.warning(f"Model size: {size_bytes / 1024 / 1024} MB")

        torch.manual_seed(998244353)
        model.init_weights()
        model = model.to(model.device)
        ddp_model = DDP(model, device_ids=[rank])

        elapses = defaultdict(list)

        for iter in range(args["num_iters"]):
            model.x = torch.randn(
                1,
                model.layer_size,
                requires_grad=True,
            ).to(
                model.device,
            )
            model.y = torch.randn(
                1,
                model.layer_size,
            ).to(
                model.device,
            )

            if rank == 0:
                logger.info(f"iter: {iter}")
                logger.info(f"input: {model.x}")
                logger.info(f"target: {model.x}")

            torch.cuda.synchronize()
            dist.barrier()
            start = get_timing_event()

            forward_start = get_timing_event()
            pred = ddp_model(model.x)
            forward_end = get_timing_event()

            loss_compute_start = get_timing_event()
            loss = model.criterion(pred, model.y)
            loss_compute_end = get_timing_event()

            backward_start = get_timing_event()
            loss.backward()
            backward_end = get_timing_event()

            update_start = get_timing_event()
            model.optimizer.step()
            model.optimizer.zero_grad()
            update_end = get_timing_event()

            torch.cuda.synchronize()
            barrier_start = get_timing_event()

            dist.barrier()
            end = get_timing_event()

            torch.cuda.synchronize()

            if rank == 0:
                weights = model.fetch_weights()
                for i, weight in enumerate(weights):
                    logger.info(f"layer: {i}, weight: {weight}")

            total_ms = start.elapsed_time(end)

            def log(key: str, elapse_ms: float):
                elapse_us = ms_to_micros(elapse_ms)
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

    save_model = args.get("save_model", False)
    if save_model:
        if rank == 0:
            model_file = f"{args['model_prefix']}.log"
            with open(model_file, "w") as f:
                weights = model.fetch_weights()
                for weight in weights:
                    f.write(f"{weight}\n")
        model_file = f"{args['model_prefix']}_{rank}.log"
        with open(model_file, "w") as f:
            weights = model.fetch_weights()
            for weight in weights:
                f.write(f"{weight.cpu()}\n")


if __name__ == "__main__":
    args = parse_args()
    run_torch_ddp(args)
