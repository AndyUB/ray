import logging
import time
from typing import Any, Dict, List, Tuple

import ray
from ..core.common import secs_to_micros
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
                actor.update.bind(grads_allreduced[j], True, i)
                for j, actor in enumerate(actors)
            ]
            outputs.extend(actors_to_updates)

        dag = MultiOutputNode(outputs)

    compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)

    BATCH_SIZE = 32

    total_elapses: List[int] = []
    for epoch in range(num_epochs):
        for actor in actors:
            ray.get(actor.init_training.remote(BATCH_SIZE))
            ray.get(actor.init_tracing.remote())

        start = time.perf_counter()
        compiled_dag.execute(None)
        end = time.perf_counter()

        if epoch > 0:
            logger.warning(f"epoch: {epoch}, elapse: {round((end - start) * 1e6)} us")
            total_elapses.append(round((end - start) * 1e6))

        for actor in actors:
            ray.get(actor.finish_tracing.remote())

    ray.get([actor.fetch_profile.remote() for actor in actors])
    actors_to_elapses = [ray.get(actor.fetch_traces.remote()) for actor in actors]
    actors_to_serializations = [elapse["serialization"] for elapse in actors_to_elapses]
    actors_to_deserializations = [
        elapse["deserialization"] for elapse in actors_to_elapses
    ]
    total_len = sum(
        [
            len(single_iter_traces)
            for single_actor_traces in actors_to_serializations
            for single_iter_traces in single_actor_traces
        ]
    ) + sum(
        [
            len(single_iter_traces)
            for single_actor_traces in actors_to_deserializations
            for single_iter_traces in single_actor_traces
        ]
    )
    unique_traces = set(
        [
            trace
            for single_actor_traces in actors_to_serializations
            for single_iter_traces in single_actor_traces
            for trace in single_iter_traces
        ]
        + [
            trace
            for single_actor_traces in actors_to_deserializations
            for single_iter_traces in single_actor_traces
            for trace in single_iter_traces
        ]
    )
    unique_len = len(unique_traces)
    logger.warning(f"total serialization traces: {total_len}, unique: {unique_len}")

    def avg_elapses(
        actors_to_elapses: List[List[List[Tuple[float, float]]]]
    ) -> List[List[float]]:
        actors_to_avg_elapses = []
        for single_actor_traces in actors_to_elapses:
            elapses_across_iters = []
            for single_iter_traces in single_actor_traces:
                total_elapse = 0
                for start, end in single_iter_traces:
                    total_elapse += end - start
                total_elapse = secs_to_micros(total_elapse)
                elapses_across_iters.append(total_elapse)
            actors_to_avg_elapses.append(elapses_across_iters)
        return actors_to_avg_elapses

    actors_to_avg_serialization_elapses = avg_elapses(actors_to_serializations)
    actors_to_avg_deserialization_elapses = avg_elapses(actors_to_deserializations)
    for i, actor_elapses in enumerate(actors_to_elapses):
        actor_elapses["serialization"] = actors_to_avg_serialization_elapses[i]
        actor_elapses["deserialization"] = actors_to_avg_deserialization_elapses[i]

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
            "serialization",
            "deserialization",
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
