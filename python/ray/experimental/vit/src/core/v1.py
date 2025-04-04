import logging
import os

import fire
from actor import TextWorker, VisionWorker
from common import random_seed
from dist import init_torch_distributed

import ray
from ray.dag.input_node import InputNode
from ray.dag.output_node import MultiOutputNode

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
    # https://github.com/mlfoundations/open_clip/blob/main/docs/model_profile.csv
    # model_name: str = "ViT-L-14",
    model_name: str = "ViT-bigG-14",
    bs_single: int = 16,
    num_tp_vision: int = 1,
    num_dp_vision: int = 3,
    num_tp_text: int = 1,
    num_dp_text: int = 1,
    num_iters: int = 50,
):
    random_seed(998244353)
    assert num_tp_vision == num_tp_text == 1

    bs_global = bs_single * max(num_dp_vision, num_dp_text)
    bs_vision = bs_global // num_dp_vision
    bs_text = bs_global // num_dp_text
    logger.info(
        f"Global batch size: {bs_global}, vision batch size: {bs_vision}, text batch size: {bs_text}"
    )

    vision_actors = [
        VisionWorker.remote(model_name, num_dp_vision, num_tp_vision, num_dp_text)
        for _ in range(num_dp_vision * num_tp_vision)
    ]
    init_torch_distributed(vision_actors, GPU_IDS)
    ray.get([worker.init_fsdp_model.remote() for worker in vision_actors])

    text_actors = [
        TextWorker.remote(model_name, num_dp_text, num_tp_text, num_dp_vision)
        for _ in range(num_dp_text * num_tp_text)
    ]
    init_torch_distributed(text_actors, GPU_IDS)
    ray.get([worker.init_fsdp_model.remote() for worker in text_actors])

    vision_params = ray.get(vision_actors[0].get_num_params.remote())
    text_params = ray.get(text_actors[0].get_num_params.remote())
    logger.info(f"Vision params: {vision_params:,}, text params: {text_params:,}")

    vision_device = ray.get(vision_actors[0].get_device_name.remote())
    text_device = ray.get(text_actors[0].get_device_name.remote())
    logger.info(f"Vision device name: {vision_device}, text device name: {text_device}")

    with InputNode() as input_node:
        # forward
        text_acts = [actor.forward.bind(input_node) for actor in text_actors]
        vision_acts = [actor.forward.bind(input_node) for actor in vision_actors]

        # reduce activations to global batch size
        reduced_text_acts = text_actors[0].reduce_activations.bind(*text_acts)
        reduced_vision_acts = vision_actors[0].reduce_activations.bind(*vision_acts)

        # scatter activations to the other dp groups
        scattered_text_acts = (
            text_actors[0]
            .scatter_activations.options(num_returns=num_dp_vision)
            .bind(reduced_text_acts)
        )
        scattered_vision_acts = (
            vision_actors[0]
            .scatter_activations.options(num_returns=num_dp_text)
            .bind(reduced_vision_acts)
        )
        if num_dp_vision == 1 and not isinstance(scattered_text_acts, list):
            scattered_text_acts = [scattered_text_acts]
        if num_dp_text == 1 and not isinstance(scattered_vision_acts, list):
            scattered_vision_acts = [scattered_vision_acts]

        # with_tensor_transport for NCCL transport
        scattered_text_acts = [
            to_vision_act.with_tensor_transport("nccl")
            for to_vision_act in scattered_text_acts
        ]
        scattered_vision_acts = [
            to_text_act.with_tensor_transport("nccl")
            for to_text_act in scattered_vision_acts
        ]

        # backward
        outputs = []
        for i, actor in enumerate(text_actors):
            outputs.append(actor.backward.bind(scattered_vision_acts[i]))
        for i, actor in enumerate(vision_actors):
            outputs.append(actor.backward.bind(scattered_text_acts[i]))

        dag = MultiOutputNode(outputs)

    dag = dag.experimental_compile(_submit_timeout=480)

    for i in range(num_iters):
        ray.get([actor.init_training.remote() for actor in text_actors])
        ray.get([actor.init_training.remote() for actor in vision_actors])

        ray.get(dag.execute((i, bs_global)))

        logger.info(f"Iteration {i} finished")

        ray.get([actor.finish_tracing.remote() for actor in text_actors])
        ray.get([actor.finish_tracing.remote() for actor in vision_actors])


if __name__ == "__main__":
    fire.Fire(main)
