import logging
from typing import Any, Dict, List

import torch

import ray
from ....core.common import get_timing_event, log_elapses_to_csv
from ....core.config import parse_args
from ....core.linear.actor import LinearActor
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce, allgather, reducescatter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)
logger.info("Welcome to Downton Abbey!")


def init_actors(args: Dict[str, Any]) -> List[LinearActor]:
    layer_size = args["layer_size"]
    num_layers = args["num_layers"]
    num_partitions = args["num_partitions"]
    num_actors = args["num_actors"]
    device = "cuda:0"
    tracing = args["tracing"]

    actor_cls = LinearActor.options(num_gpus=1)
    actors = [
        actor_cls.remote(
            layer_size=layer_size,
            num_layers=num_layers,
            num_partitions=num_partitions,
            num_actors=num_actors,
            device=device,
            tracing=tracing,
        )
        for _ in range(num_actors)
    ]

    return actors


def train(
    actors: List[LinearActor],
    num_units: int,
    num_iters: int,
    output_path: str,
    latency_prefix: str,
    save_model: bool,
    model_prefix: str,
    tracing: bool,
) -> None:
    with InputNode() as inp:
        xs = [actor.get_input.bind(inp) for actor in actors]
        for unit in range(num_units):
            shards = [actor.get_shard.bind(unit, inp) for actor in actors]
            unsharded_params = allgather.bind(shards)
            xs = [
                actor.forward.bind(unit, unsharded_param, x)
                for actor, unsharded_param, x in zip(actors, unsharded_params, xs)
            ]
        ys = [actor.get_output.bind(inp) for actor in actors]
        losses = [actor.compute_loss.bind(x, y) for actor, x, y in zip(actors, xs, ys)]
        grads = [actor.backward_loss.bind(loss) for actor, loss in zip(actors, losses)]
        reduced_grads = reducescatter.bind(grads)
        outputs = []
        updates = [
            actor.update.bind(num_units - 1, reduced_grad)
            for actor, reduced_grad in zip(actors, reduced_grads)
        ]
        outputs.extend(updates)
        for unit in reversed(range(num_units - 1)):
            shards = [actor.get_shard.bind(unit, inp) for actor in actors]
            unsharded_params = allgather.bind(shards)
            grads = [
                actor.backward.bind(unit, unsharded_param)
                for actor, unsharded_param in zip(actors, unsharded_params)
            ]
            reduced_grads = reducescatter.bind(grads)
            updates = [
                actor.update.bind(unit, reduced_grad)
                for actor, reduced_grad in zip(actors, reduced_grads)
            ]
            outputs.extend(updates)
        dag = MultiOutputNode(outputs)

    compiled_dag = dag.experimental_compile(_overlap_gpu_communication=True)

    shards_across_actors = ray.get(actors[0].init_and_shard_model.remote())
    for actor, shards in zip(actors, shards_across_actors):
        ray.get(actor.set_shards.remote(shards))

    total_elapses: List[int] = []
    for iter in range(num_iters):
        for actor in actors:
            ray.get(actor.init_training.remote())
            ray.get(actor.init_tracing.remote())

        start = get_timing_event()
        compiled_dag.execute(None)
        end = get_timing_event()
        torch.cuda.synchronize()

        elapse_ms = start.elapsed_time(end)
        elapse_us = round(elapse_ms * 1e3)

        if save_model:
            weights = ray.get(actors[0].fetch_weights.remote())
            for idx, weight in enumerate(weights):
                logger.info(f"layer: {idx}, weight: {weight}")

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

    compiled_dag.teardown()


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
