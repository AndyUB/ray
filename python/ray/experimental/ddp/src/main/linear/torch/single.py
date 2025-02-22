import logging
import time
from typing import Any, Dict, List, Tuple

import torch

from ....core.config import parse_args
from ....core.linear.model import BucketParameter

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)
logger.info("Welcome to Downton Abbey!")


def init_models(args: Dict[str, Any]) -> List[BucketParameter]:
    layer_size = args["layer_size"]
    num_layers = args["num_layers"]
    device = "cuda:0"
    num_partitions = args["num_partitions"]

    models = [
        BucketParameter(
            layer_size=layer_size,
            num_layers=num_layers // num_partitions,
            device=device,
        )
        for _ in range(num_partitions)
    ]

    return models


def init_weights(models: List[BucketParameter]) -> None:
    torch.manual_seed(998244353)
    for model in models:
        model.init_weights()
        model = model.to(model.device)


def init_training(models: List[BucketParameter]) -> None:
    # Generate input for first model and target for last model
    models[0].x = torch.randn(
        1,
        models[0].layer_size,
        requires_grad=True,
    ).to(
        models[0].device,
    )
    models[-1].y = torch.randn(
        1,
        models[-1].layer_size,
    ).to(
        models[-1].device,
    )


def forward(
    models: List[BucketParameter],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    intermediates = []
    input = models[0].x
    for i, model in enumerate(models):
        pred = model.forward(input)
        if i < len(models) - 1:
            input = pred.detach().requires_grad_(True)
        else:
            input = pred
        intermediates.append((pred, input))
    return intermediates


def backward(
    models: List[BucketParameter],
    intermediates: List[Tuple[torch.Tensor, torch.Tensor]],
    idx: int,
) -> None:
    if idx == len(models) - 1:
        loss = models[idx].criterion(
            intermediates[idx][0],
            models[idx].y,
        )
        pred = None
        grad = None
    else:
        loss = None
        pred, input = intermediates[idx]
        grad = input.grad
    models[idx].backward(
        loss=loss,
        pred=pred,
        grad=grad,
    )


def update(
    models: List[BucketParameter],
    idx: int,
) -> None:
    models[idx].update(None, False)


def train_cot(
    models: List[BucketParameter],
    num_iters: int,
    model_file: str,
) -> None:
    init_weights(models)

    for iter in range(num_iters):
        start = time.perf_counter()

        init_training(models)

        logger.info(f"iter: {iter}")
        logger.info(f"input: {models[0].x}")
        logger.info(f"target: {models[-1].y}")

        # Forward pass through all models
        intermediates = forward(models)
        logger.info(f"prediction: {intermediates[-1][0]}")

        # Propagate gradients backward through intermediate models
        for i in reversed(range(len(intermediates))):
            backward(
                models,
                intermediates,
                i,
            )

        for i in reversed(range(len(intermediates))):
            update(
                models,
                i,
            )

        end = time.perf_counter()

        # Log updated weights
        for model_idx, model in enumerate(models):
            weights = model.fetch_weights()
            for layer_idx, layer_weight in enumerate(weights):
                logger.info(
                    f"model: {model_idx}, layer: {layer_idx}, weight: {layer_weight}"
                )

        if iter > 0:
            logger.warning(f"iter: {iter}, elapse: {round((end - start) * 1e6)} us")

    with open(model_file, "w") as f:
        for model in models:
            weights = model.fetch_weights()
            for weight in weights:
                f.write(f"{weight}\n")


def main(args: Dict[str, Any]) -> None:
    models = init_models(args)

    train_cot(
        models,
        args["num_iters"],
        args["model_file"],
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
