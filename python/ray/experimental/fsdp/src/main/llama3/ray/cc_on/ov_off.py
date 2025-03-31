import logging
from typing import Any, Dict, List, Optional, Tuple

import ray
from .....core.common import get_end_time, get_start_time, log_elapses_to_csv
from .....core.config import parse_args
from .....core.llama3.actor import LlamaActor
from .....core.llama3.model import LLAMA_DEBUG as LLAMA
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allgather, reducescatter

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def init_actors(args: Dict[str, Any]) -> List[LlamaActor]:
    model_args = LLAMA
    batch_size = args["batch_size"]
    seq_len = args["seq_len"]
    num_partitions = args["num_partitions"]
    num_actors = args["num_actors"]
    tracing = args["tracing"]

    actor_cls = LlamaActor.options(num_gpus=1)
    actors = [
        actor_cls.remote(
            model_args,
            batch_size=batch_size,
            seq_len=seq_len,
            num_partitions=num_partitions,
            num_actors=num_actors,
            tracing=tracing,
        )
        for _ in range(num_actors)
    ]

    return actors


def train(
    actors: List[LlamaActor],
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
        shards = [actor.get_shard.bind(0, inp) for actor in actors]
        params = allgather.bind(shards)
        for idx in range(num_units):
            if idx < num_units - 1:
                shards_pf = [actor.get_shard.bind(idx + 1, inp) for actor in actors]
                params_pf = allgather.bind(shards_pf)

            inputs = [
                actor.forward.bind(idx, param, input)
                for actor, param, input in zip(actors, params, inputs)
            ]

            if idx < num_units - 1:
                params = params_pf

        targets = [actor.get_target.bind(inp) for actor in actors]
        losses = [
            actor.compute_loss.bind(output, target)
            for actor, output, target in zip(actors, inputs, targets)
        ]

        unit_to_grads = []
        for idx in reversed(range(-1, num_units)):
            if idx + 1 < num_units:
                # Reduce grads for unit (idx + 1).
                reduced_grads = reducescatter.bind(grads)
                unit_to_grads.append(reduced_grads)

            if idx - 1 >= 0:
                # Prefetch params for unit (idx - 1).
                shards_pf = [actor.get_shard.bind(idx - 1, inp) for actor in actors]
                params_pf = allgather.bind(shards_pf)

            # [TODO] Timing for backward is not accurate since it is a future.
            if idx == num_units - 1:
                # Backward grads for unit (num_units - 1).
                grads = [
                    actor.backward_loss.bind(loss)
                    for actor, loss in zip(actors, losses)
                ]
            elif idx >= 0:
                # Backward grads for unit (idx).
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

            if idx - 1 >= 0:
                # Set params for unit (idx - 1).
                params = params_pf

        unit_to_grads = unit_to_grads[::-1]
        updates = []
        for idx in reversed(range(num_units)):
            grads = unit_to_grads[idx]
            updates.extend(
                [
                    actor.update.bind(idx, grad, True)
                    for actor, grad in zip(actors, grads)
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
    metrics = LlamaActor.get_metrics(tracing)
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
