import logging
import os
import time

import torch
from fairscale.nn.model_parallel.initialize import (
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from .....core.llama3.actor import _Actor_V2 as Actor
from .....core.llama3.model import LLAMA_1B

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "25670"

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    if not model_parallel_is_initialized():
        model_parallel_size = 1
        initialize_model_parallel(model_parallel_size)

    local_rank = 0
    torch.cuda.set_device(local_rank)
    torch.manual_seed(998244353)

    model_args = LLAMA_1B
    actor = Actor(model_args)
    logger.info(actor.model)

    n_iters = 3

    for _ in range(n_iters):
        actor.init_training()

        fw_start = time.perf_counter()
        logits = actor.forward(None)
        fw_end = time.perf_counter()
        logger.info(f"Forward time: {round((fw_end - fw_start) * 1e3)} ms")

        bw_start = time.perf_counter()
        actor.backward_aio(logits)
        bw_end = time.perf_counter()
        logger.info(f"Backward time: {round((bw_end - bw_start) * 1e3)} ms")

        upd_start = time.perf_counter()
        actor.update_aio(None)
        upd_end = time.perf_counter()
        logger.info(f"Update time: {round((upd_end - upd_start) * 1e3)} ms")

        logger.info("")


if __name__ == "__main__":
    main()
