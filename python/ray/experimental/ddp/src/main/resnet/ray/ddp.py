import logging
from typing import Any, Dict, List

import torch

import ray
from ....core.common import get_timing_event, log_elapses_to_csv
from ....core.config import parse_args
from ....core.resnet.actor import ResnetActor
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)
logger.info("Welcome to Downton Abbey!")


def init_actors(args: Dict[str, Any]) -> List[ResnetActor]:
    num_partitions = args["num_partitions"]
    num_actors = args["num_actors"]
    device = "cuda:0"
    tracing = args["tracing"]

    actor_cls = ResnetActor.options(num_gpus=1)
    actors = [
        actor_cls.remote(
            rank=i,
            num_partitions=num_partitions,
            num_actors=num_actors,
            device=device,
            tracing=tracing,
        )
        for i in range(num_actors)
    ]

    return actors


def train(
    actors: List[ResnetActor],
    num_partitions: int,
    num_iters: int,
    output_path: str,
    latency_prefix: str,
    save_model: bool,
    model_prefix: str,
    tracing: bool,
) -> None:
    with InputNode() as inp:
        actors_to_forwards = [actor.forward.bind(inp) for actor in actors]
        actors_to_backwards = actors_to_forwards
        outputs = []

        actors_to_backwards = [
            actor.backward.bind(actors_to_backwards[j], num_partitions - 1)
            for j, actor in enumerate(actors)
        ]
        for i in reversed(range(num_partitions)):
            grads_allreduced = allreduce.bind(actors_to_backwards)
            if i > 0:
                actors_to_backwards = [
                    actor.backward.bind(actors_to_backwards[j], i - 1)
                    for j, actor in enumerate(actors)
                ]
            actors_to_updates = [
                actor.update.bind(grads_allreduced[j], True, i)
                for j, actor in enumerate(actors)
            ]
            outputs.extend(actors_to_updates)

        dag = MultiOutputNode(outputs)

    compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)

    BATCH_SIZE = 32

    total_elapses: List[int] = []
    for iter in range(num_iters):
        for actor in actors:
            ray.get(actor.init_training.remote(BATCH_SIZE))
            ray.get(actor.init_tracing.remote())

        start = get_timing_event()
        compiled_dag.execute(None)
        end = get_timing_event()
        torch.cuda.synchronize()

        elapse_ms = start.elapsed_time(end)
        elapse_us = round(elapse_ms * 1e3)

        if iter > 0:
            logger.warning(f"iter: {iter}, elapse: {elapse_us} us")
            total_elapses.append(elapse_us)

        for actor in actors:
            ray.get(actor.finish_tracing.remote())

    actors_to_elapses = [ray.get(actor.fetch_traces.remote()) for actor in actors]
    for actor_elapses in actors_to_elapses:
        actor_elapses["total"] = total_elapses
    if not tracing:
        metrics = [
            "total",
            "actor.total",
        ]
    else:
        metrics = [
            "total",
            "actor.total",
            "fw.total",
            "bw.total",
            "bw.backward",
            "bw.others",
            "bw.update",
        ]
    log_elapses_to_csv(
        actors_to_elapses,
        output_path,
        latency_prefix,
        metrics,
    )

    if save_model:
        model_file = f"{model_prefix}.log"
        with open(model_file, "w") as f:
            for weight in weights:
                f.write(f"{weight}\n")
        for i, actor in enumerate(actors):
            weights = ray.get(actor.fetch_weights.remote())
            model_file = f"{model_prefix}_{i}.log"
            with open(model_file, "w") as f:
                for weight in weights:
                    f.write(f"{weight}\n")


def main(args: Dict[str, Any]) -> None:
    ray.init()

    actors = init_actors(args)

    train(
        actors,
        args["num_partitions"],
        args["num_iters"],
        args["output_path"],
        args["latency_prefix"],
        args.get("save_model", False),
        args["model_prefix"],
        args["tracing"],
    )

    for actor in actors:
        ray.kill(actor)
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    main(args)
