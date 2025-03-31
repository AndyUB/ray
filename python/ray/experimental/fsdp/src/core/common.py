import csv
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def get_start_time() -> float:
    return time.perf_counter()


def get_end_time(sync: bool = True) -> float:
    if sync:
        torch.cuda.synchronize()
    return time.perf_counter()


def get_timing_event_torch() -> torch.cuda.Event:
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def secs_to_micros(secs: float) -> int:
    """
    Converts seconds to microseconds.
    """
    return round(secs * 1e6)


def millis_to_micros(millis: float) -> int:
    """
    Converts milliseconds to microseconds.
    """
    return round(millis * 1e3)


def log_elapses_to_csv(
    ranks_to_elapses: List[Dict[str, Any]],
    output_path: str,
    latency_prefix: str,
    metrics: List[str],
    aliases: Optional[List[str]] = None,
    warmup: float = 0.2,
) -> None:
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    if aliases is None:
        aliases = [None] * len(metrics)

    # Process each rank's data
    for idx, metric_values in enumerate(ranks_to_elapses):
        output_file = f"{output_path}/{latency_prefix}_rank{idx}.csv"

        for metric in metrics:
            assert metric in metric_values
            metric_values[metric] = metric_values[metric][
                int(warmup * len(metric_values[metric])) :
            ]

        # Calculate statistics for each metric
        total_mean = (
            np.mean(metric_values["actor.total"]) if metric_values["actor.total"] else 0
        )

        # Write statistics to CSV file
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["name", "mean", "std", "cv", "percent"])

            for metric, alias in zip(metrics, aliases):
                if alias is None:
                    alias = metric
                values = np.array(metric_values[metric])
                mean = np.mean(values)
                std = np.std(values)
                cv = std / mean * 100 if mean > 0 else 0
                percent = (mean / total_mean * 100) if total_mean > 0 else 0

                writer.writerow(
                    [
                        alias,
                        round(mean),
                        round(std),
                        round(cv, 1),
                        round(percent, 1),
                    ]
                )
