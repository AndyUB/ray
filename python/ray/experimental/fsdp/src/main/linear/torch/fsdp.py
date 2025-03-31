import functools
import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, ShardedStateDictConfig
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from torch.distributed.fsdp.api import BackwardPrefetch
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.profiler import profile, ProfilerActivity

from ....core.common import get_timing_event_torch, log_elapses_to_csv, millis_to_micros
from ....core.config import parse_args
from ....core.linear.model import LinearModel
from ....core.linear.actor import BucketParameter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)
logger.info("Welcome to Downton Abbey!")


class LinearModel(torch.nn.Module):
    def __init__(
        self,
        layer_size: int,
        num_layers: int,
        num_units: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        assert num_layers % num_units == 0

        self.layer_size = layer_size
        self.num_layers = num_layers
        self.num_units = num_units
        self.device = device
        self.bparams = torch.nn.ModuleList(
            [
                BucketParameter(
                    layer_size,
                    num_layers // num_units,
                    device,
                )
                for _ in range(num_units)
            ]
        )

        self.x = None
        self.y = None
        self.criterion = torch.nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for bparam in self.bparams:
            x = bparam(x)
        return x

    def init_weights(self) -> None:
        for bparam in self.bparams:
            bparam.init_weights()


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
        "loss.compute",
        "bw.total",
        "bw.upd",
        "barrier",
    ]
    aliases = [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
    log_elapses_to_csv(
        ranks_to_elapses_list,
        output_path,
        latency_prefix,
        metrics,
        aliases,
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

        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dist.init_process_group(
            "nccl", rank=rank, world_size=world_size, device_id=device
        )
        model = LinearModel(
            args["layer_size"],
            args["num_layers"],
            args["num_partitions"],
            device,
        )
        size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        logger.warning(f"Model size: {size_bytes / 1024 / 1024} MB")

        # torch.manual_seed(998244353)
        seed = 998244353
        model.init_weights()
        model = model.to(model.device)
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=functools.partial(
                lambda_auto_wrap_policy,
                lambda_fn=lambda p: isinstance(p, BucketParameter),
            ),
            device_id=device,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            forward_prefetch=True,
        )
        optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=1e-3)
        if rank == 0:
            logger.warning(f"fsdp_model: {fsdp_model}")

        elapses = defaultdict(list)

        def exec_iter():
            torch.manual_seed(seed)
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
                logger.warning(f"iter: {iter}")

            torch.cuda.synchronize()
            dist.barrier()
            start = get_timing_event_torch()

            fw_start = get_timing_event_torch()
            pred = fsdp_model(model.x)
            fw_end = get_timing_event_torch()

            loss_start = get_timing_event_torch()
            loss = model.criterion(pred, model.y)
            loss_end = get_timing_event_torch()

            bw_start = get_timing_event_torch()
            loss.backward()
            bw_end = get_timing_event_torch()

            bw_upd_start = get_timing_event_torch()
            optimizer.step()
            optimizer.zero_grad()
            bw_upd_end = get_timing_event_torch()

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
                log("loss.compute", loss_start.elapsed_time(loss_end))
                log("bw.grad", bw_start.elapsed_time(bw_end))
                log("bw.upd", bw_upd_start.elapsed_time(bw_upd_end))
                log("barrier", barrier_start.elapsed_time(end))
                logger.warning("")

        if args["profile"]:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for iter in range(args["num_iters"]):
                    exec_iter()
                    seed += 1
                    prof.step()

            num_layers = args["num_layers"]
            layer_size = args["layer_size"]
            num_partitions = args["num_partitions"]
            num_iters = args["num_iters"]
            prof.export_chrome_trace(
                os.path.join(
                    args["output_path"],
                    f"rank{rank}_"
                    f"{num_layers}layer_"
                    f"{layer_size}size_"
                    f"{num_partitions}unit_"
                    f"{num_iters}it.json",
                )
            )
        else:
            for iter in range(args["num_iters"]):
                exec_iter()
                seed += 1

        save_model = args.get("save_model", False)
        if save_model:
            # dist.barrier()
            # save_policy = ShardedStateDictConfig()
            # model_file = args["model_file"]
            # with FSDP.state_dict_type(
            #     fsdp_model, StateDictType.SHARDED_STATE_DICT, save_policy
            # ):
            #     state_dict = fsdp_model.state_dict()
            #     torch.save(state_dict, f"{model_file}_rank{rank}.pt")
            # dist.barrier()

            dist.barrier()
            full_state_dict_config = FullStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            )
            with FSDP.state_dict_type(
                fsdp_model, StateDictType.FULL_STATE_DICT, full_state_dict_config
            ):
                state_dict = fsdp_model.state_dict()
                model_file = args.get("model_file", "torch_fsdp_model.pt")
                if rank == 0:
                    torch.save(state_dict, model_file)
                dist.barrier()

    finally:
        dist.destroy_process_group()

    ranks_to_elapses[rank] = elapses

    # if save_model:
    #     if rank == 0:
    #         model_file = f"{args['model_prefix']}.log"
    #         with open(model_file, "w") as f:
    #             weights = model.fetch_weights()
    #             for weight in weights:
    #                 f.write(f"{weight}\n")
    #     model_file = f"{args['model_prefix']}_{rank}.log"
    #     with open(model_file, "w") as f:
    #         weights = model.fetch_weights()
    #         for weight in weights:
    #             f.write(f"{weight.cpu()}\n")


if __name__ == "__main__":
    args = parse_args()
    run_torch_fsdp(args)
