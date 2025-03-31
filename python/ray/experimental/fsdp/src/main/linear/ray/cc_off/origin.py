import logging
from typing import Any, Dict, List, Optional, Tuple

import ray
from .....core.common import get_end_time, get_start_time, log_elapses_to_csv
from .....core.config import parse_args
from .....core.linear.actor import LinearActor
from ray.dag import InputNode, MultiOutputNode

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)
logger.info("Welcome to Downton Abbey!")


def init_actors(args: Dict[str, Any]) -> List[LinearActor]:
    layer_size = args["layer_size"]
    num_layers = args["num_layers"]
    num_units = args["num_partitions"]
    num_actors = args["num_actors"]
    device = "cuda:0"
    tracing = args["tracing"]

    actor_cls = LinearActor.options(num_gpus=1)
    actors = [
        actor_cls.remote(
            layer_size=layer_size,
            num_layers=num_layers,
            num_units=num_units,
            num_actors=num_actors,
            device=device,
            tracing=tracing,
        )
        for _ in range(num_actors)
    ]

    return actors


def get_metrics_aliases(tracing: bool) -> Tuple[List[str], List[Optional[str]]]:
    if not tracing:
        metrics = [
            "total",
            "actor.total",
        ]
        alias = [
            "!total",
            None,
        ]
    else:
        metrics = [
            "total",
            "actor.total",
            "fw.total",
            "loss.total",
            "bw.total",
            "bw.loss",
            "bw.grad.pre",
            "bw.grad.intra",
            "bw.grad.post",
            "bw.grad.wo.loss_upd",
            "bw.upd",
        ]
        alias = [
            "!total",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]
    return metrics, alias


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
        inputs = [actor.get_input.bind(inp) for actor in actors]
        for idx in range(num_units):
            shards = [actor.get_shard.bind(idx, inp) for actor in actors]
            params = shards
            inputs = [
                actor.forward.bind(idx, param, input)
                for actor, param, input in zip(actors, params, inputs)
            ]

        targets = [actor.get_target.bind(inp) for actor in actors]
        losses = [
            actor.compute_loss.bind(output, target)
            for actor, output, target in zip(actors, inputs, targets)
        ]

        grads = [actor.backward_loss.bind(loss) for actor, loss in zip(actors, losses)]
        reduced_grads = grads
        updates = [
            actor.update.bind(num_units - 1, grad, True)
            for actor, grad in zip(actors, reduced_grads)
        ]

        for idx in reversed(range(num_units - 1)):
            shards = [actor.get_shard.bind(idx, inp) for actor in actors]
            params = shards
            bw_pres = [
                actor.backward_pre.bind(idx, param)
                for actor, param in zip(actors, params)
            ]
            bw_intras = [
                actor.backward_intra.bind(idx, param, pre)
                for actor, param, pre in zip(actors, params, bw_pres)
            ]
            grads = [
                actor.backward_post.bind(idx, param, intra)
                for actor, param, intra in zip(actors, params, bw_intras)
            ]
            reduced_grads = grads
            updates.extend(
                [
                    actor.update.bind(idx, grad, True)
                    for actor, grad in zip(actors, reduced_grads)
                ]
            )

        dag = MultiOutputNode(updates)

    compiled_dag = dag.experimental_compile()
    actor_to_shards = ray.get(actors[0].init_and_shard_model.remote())
    for actor, shards in zip(actors, actor_to_shards):
        ray.get(actor.set_shards.remote(shards))

    total_elapses: List[int] = []
    for iter in range(num_iters):
        for actor in actors:
            ray.get(actor.init_training.remote())
            ray.get(actor.init_tracing.remote())

        start = get_start_time()
        compiled_dag.execute(None)
        end = get_end_time()
        elapse_us = round((end - start) * 1e6)

        if save_model:
            for i, actor in enumerate(actors):
                weights = ray.get(actor.fetch_weights.remote())
                for idx, weight in enumerate(weights):
                    logger.info(f"actor: {i}, layer: {idx}, shard: {weight}")

        if iter > 0:
            logger.warning(f"iter: {iter}, elapse: {elapse_us} us")
            total_elapses.append(elapse_us)

        for actor in actors:
            ray.get(actor.finish_tracing.remote())

    actors_to_elapses = [ray.get(actor.fetch_traces.remote()) for actor in actors]
    for actor_elapses in actors_to_elapses:
        actor_elapses["total"] = total_elapses
    metrics, alias = get_metrics_aliases(tracing)
    log_elapses_to_csv(
        actors_to_elapses,
        output_path,
        latency_prefix,
        metrics,
        alias,
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
