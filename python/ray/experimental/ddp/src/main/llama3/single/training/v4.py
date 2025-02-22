import logging

import torch

from .....core.common import get_end_time, get_start_time
from .....core.llama3.actor import _Actor_V4 as Actor
from .....core.llama3.model import LLAMA_1B

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(998244353)

    model_args = LLAMA_1B
    actor = Actor(model_args)
    logger.info(actor.model)

    n_iters = 10
    n_warmup_iters = int(n_iters * 0.2)

    for i in range(n_iters):
        actor.init_training()

        fw_start = get_start_time()
        logits = actor.forward(None)
        fw_end = get_end_time()
        if i >= n_warmup_iters:
            logger.info(f"forward: {round((fw_end - fw_start) * 1e3)} ms")

        bw_start = get_start_time()
        for j in reversed(range(len(actor.bparams))):
            bw_j_start = get_start_time()
            actor.backward(None, j)
            bw_j_end = get_end_time()
            if i >= n_warmup_iters:
                logger.info(
                    f"backward partition {j}: {round((bw_j_end - bw_j_start) * 1e3)} ms"
                )
        bw_end = get_end_time()
        if i >= n_warmup_iters:
            logger.info(f"backward: {round((bw_end - bw_start) * 1e3)} ms")

        copy_start = get_start_time()
        actor.copy_aio(None)
        copy_end = get_end_time()
        if i >= n_warmup_iters:
            logger.info(f"copy: {round((copy_end - copy_start) * 1e3)} ms")

        step_start = get_start_time()
        actor.step_aio(None)
        step_end = get_end_time()
        if i >= n_warmup_iters:
            logger.info(f"step: {round((step_end - step_start) * 1e3)} ms")
            logger.info(f"backward and update: {round((step_end - bw_start) * 1e3)} ms")
            logger.info(f"iter {i}: {round((step_end - fw_start) * 1e3)} ms")
            logger.info("")


if __name__ == "__main__":
    main()
