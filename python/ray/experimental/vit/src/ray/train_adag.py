import fire
from .adag_workers import TextWorker, VisionWorker
from .multi_process_group import initialize_dist_group
from .utils import get_logger, random_seed

import ray
from ray.dag.input_node import InputNode
from ray.dag.output_node import MultiOutputNode
from ray.util.accelerators import NVIDIA_TESLA_T4

# supported models:
# https://github.com/mlfoundations/open_clip/blob/main/docs/model_profile.csv


def main(
    model_name: str = "ViT-L-14",
    batch_size: int = 16,
    vision_tp_size: int = 1,
    vision_dp_size: int = 2,
    # Barbell
    # vision_dp_size: int = 1,
    text_tp_size: int = 1,
    text_dp_size: int = 2,
    # Barbell
    # text_dp_size: int = 1,
    seed: int = 123,
    steps: int = 50,
):

    random_seed(seed)
    logger = get_logger()

    assert vision_tp_size == text_tp_size == 1, "Only support tp size 1 for now"

    global_batch_size = batch_size * max(vision_dp_size, text_dp_size)
    vision_batch_size = global_batch_size // vision_dp_size
    text_batch_size = global_batch_size // text_dp_size
    logger.info(
        f"Global batch size: {global_batch_size}, "
        f"Vision batch size: {vision_batch_size}, "
        f"Text batch size: {text_batch_size}"
    )

    logger.info("Start aDAG compiling...")

    # change to the available accelerator type
    vision_workers = [
        # VisionWorker.options(accelerator_type=NVIDIA_TESLA_T4).remote(
        VisionWorker.remote(model_name, vision_dp_size, vision_tp_size, text_dp_size)
        for _ in range(vision_dp_size * vision_tp_size)
    ]
    initialize_dist_group(vision_workers)
    ray.get([worker.init_parallel_strategy.remote() for worker in vision_workers])

    # change to the available accelerator type
    text_workers = [
        # TextWorker.options(accelerator_type=NVIDIA_TESLA_T4).remote(
        TextWorker.remote(model_name, text_dp_size, text_tp_size, vision_dp_size)
        for _ in range(text_dp_size * text_tp_size)
    ]
    initialize_dist_group(text_workers)
    ray.get([worker.init_parallel_strategy.remote() for worker in text_workers])

    # Get model parameters
    vision_params = ray.get(vision_workers[0].get_num_params.remote())
    text_params = ray.get(text_workers[0].get_num_params.remote())
    logger.info(f"Vision params: {vision_params:,}\tText params: {text_params:,}")

    vision_device_name = ray.get(vision_workers[0].get_device_name.remote())
    text_device_name = ray.get(text_workers[0].get_device_name.remote())
    logger.info(
        f"Vision device name: {vision_device_name}\tText device name: {text_device_name}"
    )

    outputs = []
    with InputNode() as input_node:

        # forward
        text_activations = [worker.forward.bind(input_node) for worker in text_workers]
        vision_activations = [
            worker.forward.bind(input_node) for worker in vision_workers
        ]

        # reduce activations to global batch size
        reduced_text_activations = text_workers[0].reduce_activations.bind(
            *text_activations
        )
        reduced_vision_activations = vision_workers[0].reduce_activations.bind(
            *vision_activations
        )

        # scatter activations to the other dp groups
        scattered_text_activations = (
            text_workers[0]
            .scatter_activations.options(num_returns=vision_dp_size)
            .bind(reduced_text_activations)
        )
        # Barbell
        if vision_dp_size == 1:
            scattered_text_activations = [scattered_text_activations]

        scattered_vision_activations = (
            vision_workers[0]
            .scatter_activations.options(num_returns=text_dp_size)
            .bind(reduced_vision_activations)
        )
        # Barbell
        if text_dp_size == 1:
            scattered_vision_activations = [scattered_vision_activations]

        # with_tensor_transport for NCCL transport
        scattered_text_activations = [
            to_vision_activation.with_tensor_transport("nccl")
            for to_vision_activation in scattered_text_activations
        ]

        scattered_vision_activations = [
            to_text_activation.with_tensor_transport("nccl")
            for to_text_activation in scattered_vision_activations
        ]

        outputs = []
        for i, worker in enumerate(text_workers):
            outputs.append(worker.backward.bind(scattered_vision_activations[i]))

        for i, worker in enumerate(vision_workers):
            outputs.append(worker.backward.bind(scattered_text_activations[i]))

        dag = MultiOutputNode(outputs)
    dag = dag.experimental_compile(_submit_timeout=480)
    print("Done compiling.")

    for i in range(steps):
        ray.get(dag.execute((i, global_batch_size)))

        if (i + 1) % 10 == 0:
            logger.info(f"Done {i+1}")


if __name__ == "__main__":
    fire.Fire(main)
