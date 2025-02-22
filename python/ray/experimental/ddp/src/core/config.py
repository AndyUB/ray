import argparse
from dataclasses import dataclass
from typing import Any, Dict

import torch


def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer-size",
        type=int,
        # required=True,
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        # required=True,
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--latency-prefix",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
    )
    parser.add_argument(
        "--model-file",
        type=str,
    )
    parser.add_argument(
        "--model-prefix",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tracing",
        action="store_true",
    )
    parser.add_argument(
        "--mode",
        type=str,
    )
    args = parser.parse_args()
    args = vars(args)
    return args


@dataclass
class Config:
    """Configuration for the demo DDP model."""

    # Model config.
    dtype: torch.dtype
    # The layer is a square (n * n).
    layer_size: int
    num_layers: int

    # Training config.
    learning_rate: float
    num_actors: int
    num_iters: int

    # Output path.
    output_path: str
    # Output prefix.
    output_prefix: str
    # Check correctness.
    check_correctness: bool
    # Check performance breakdown.
    check_breakdown: bool


def parse_config() -> Config:
    """
    Parse the command line arguments and construct the corresponding configuration.

    Returns:
        Configuration for the demo DDP model.
    """

    str_to_dtype = {
        "float32": torch.float32,
        "float": torch.float,  # alias for float32
        "float64": torch.float64,
        "double": torch.double,  # alias for float64
        "float16": torch.float16,
        "half": torch.half,  # alias for float16
    }
    parser = argparse.ArgumentParser(
        description="DDP demo (ray DDP vs torch vs torch DDP)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=list(str_to_dtype.keys()),
        required=True,
        help="data type of tensors",
    )
    parser.add_argument(
        "--layer-size",
        type=int,
        required=True,
        help="size of a layer (each layer is a square)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        required=True,
        help="number of layers",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        required=True,
        help="learning rate",
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        required=True,
        help="number of actors",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        required=True,
        help="number of iterations",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="output file path",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        required=True,
        help="output file prefix",
    )
    parser.add_argument(
        "--check-correctness",
        action="store_true",
        help="whether to check correctness",
    )
    parser.add_argument(
        "--check-breakdown",
        action="store_true",
        help="whether to print performance breakdown",
    )
    args = parser.parse_args()
    config = Config(
        dtype=str_to_dtype[args.dtype],
        layer_size=args.layer_size,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        num_actors=args.num_actors,
        num_iters=args.num_iters,
        output_path=args.output_path,
        output_prefix=args.output_prefix,
        check_correctness=args.check_correctness,
        check_breakdown=args.check_breakdown,
    )
    return config
