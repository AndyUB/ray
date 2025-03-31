import logging

import torch

from ....core.common import get_end_time, get_start_time
from ....core.llama3.actor import _Actor_V5 as Actor
from ....core.llama3.model import LLAMA_DEBUG as LLAMA

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(998244353)

    model_args = LLAMA
    actor = Actor(model_args)
    logger.info(actor.model)

    n_iters = 5
    n_warmup_iters = int(n_iters * 0.2)

    for i in range(n_iters):
        actor.init_training()

        fw_start = get_start_time()
        logits = actor.forward(None)
        fw_end = get_end_time()
        if i >= n_warmup_iters:
            logger.info(f"forward: {round((fw_end - fw_start) * 1e3)} ms")

        bw_start = get_start_time()
        actor.backward(None)
        bw_end = get_end_time()
        if i >= n_warmup_iters:
            logger.info(f"backward: {round((bw_end - bw_start) * 1e3)} ms")

        upd_start = get_start_time()
        actor.update(None)
        upd_end = get_end_time()
        if i >= n_warmup_iters:
            logger.info(f"update: {round((upd_end - upd_start) * 1e3)} ms")
            logger.info(f"backward and update: {round((upd_end - bw_start) * 1e3)} ms")
            logger.info(f"iter {i}: {round((upd_end - fw_start) * 1e3)} ms")
            logger.info("")


if __name__ == "__main__":
    main()
