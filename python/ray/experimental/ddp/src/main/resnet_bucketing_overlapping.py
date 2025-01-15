import logging
import time
from typing import Any, Dict, List

import ray
from ..core.common import log_elapses_to_csv
from ..core.config import parse_args
from ..core.mp.actor_resnet import ResnetActor
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)
logger.info("Welcome to Downton Abbey!")


def init_actors(args: Dict[str, Any]) -> List[ResnetActor]:
    num_models = args["num_models"]
    num_actors = args["num_actors"]
    device = "cuda:0"
    check_tracing = args["check_tracing"]

    actor_cls = ResnetActor.options(num_gpus=1)
    actors = [
        actor_cls.remote(
            rank=i,
            num_models=num_models,
            num_actors=num_actors,
            device=device,
            check_tracing=check_tracing,
        )
        for i in range(num_actors)
    ]

    return actors


def train_cot(
    actors: List[ResnetActor],
    num_models: int,
    num_epochs: int,
    output_path: str,
    latency_prefix: str,
    save_model: bool,
    model_prefix: str,
    check_tracing: bool,
) -> None:
    with InputNode() as inp:
        actors_to_forwards = [actor.forward.bind(inp) for actor in actors]
        actors_to_backwards = actors_to_forwards
        outputs = []

        actors_to_backwards = [
            actor.backward.bind(actors_to_backwards[j], num_models - 1)
            for j, actor in enumerate(actors)
        ]
        for i in reversed(range(num_models)):
            grads_allreduced = allreduce.bind(actors_to_backwards)
            if i > 0:
                actors_to_backwards = [
                    actor.backward.bind(actors_to_backwards[j], i - 1)
                    for j, actor in enumerate(actors)
                ]
            actors_to_updates = [
                actor.update.bind(grads_allreduced[j], False, i)
                for j, actor in enumerate(actors)
            ]
            outputs.extend(actors_to_updates)

        dag = MultiOutputNode(outputs)

    compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)

    BATCH_SIZE = 32

    total_elapses: List[int] = []
    ray.get([actor.start_profile.remote() for actor in actors])
    for epoch in range(num_epochs):
        for actor in actors:
            ray.get(actor.init_training.remote(BATCH_SIZE))
            ray.get(actor.init_tracing.remote())

        time.sleep(3)

        start = time.perf_counter()
        compiled_dag.execute(None)
        end = time.perf_counter()

        if epoch > 0:
            logger.warning(f"epoch: {epoch}, elapse: {round((end - start) * 1e6)} us")
            total_elapses.append(round((end - start) * 1e6))

        for actor in actors:
            ray.get(actor.finish_tracing.remote())

        time.sleep(3)

    actors_to_elapses = [ray.get(actor.fetch_traces.remote()) for actor in actors]
    ray.get([actor.fetch_profile.remote() for actor in actors])
    for actor_elapses in actors_to_elapses:
        actor_elapses["total"] = total_elapses
    if not check_tracing:
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

    time.sleep(1)


def main(args: Dict[str, Any]) -> None:
    ray.init()

    actors = init_actors(args)

    train_cot(
        actors,
        args["num_models"],
        args["num_epochs"],
        args["output_path"],
        args["latency_prefix"],
        args.get("save_model", False),
        args["model_prefix"],
        args["check_tracing"],
    )

    for actor in actors:
        ray.kill(actor)
    ray.shutdown()


if __name__ == "__main__":
    args = parse_args()
    main(args)
