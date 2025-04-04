import logging

import fire
import os
from actor import WorkerV3 as Worker
from dist import init_torch_distributed

import ray

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


# [HACK]
CUDA_VISIBLE_DEVICES = "2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
GPU_IDS = [int(device_id) for device_id in CUDA_VISIBLE_DEVICES.split(",")]


def main(
    # model_name: str = "ViT-L-14",
    model_name: str = "ViT-bigG-14",
    bs_single: int = 16,
    num_dp_vision: int = 3,
    num_dp: int = 4,
    num_iters: int = 50,
):
    bs_global = bs_single * num_dp_vision

    actors = [Worker.remote(model_name, i, num_dp) for i in range(num_dp)]
    init_torch_distributed(actors, GPU_IDS)
    ray.get([actor.init_fsdp_model.remote() for actor in actors])

    for i in range(num_iters):
        ray.get([actor.init_training.remote() for actor in actors])

        ray.get([actor.forward.remote((i, bs_global)) for actor in actors])
        ray.get([actor.backward.remote() for actor in actors])

        logger.info(f"Iteration {i} finished")
        ray.get([actor.finish_tracing.remote() for actor in actors])


if __name__ == "__main__":
    fire.Fire(main)
