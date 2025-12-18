import time

import torch


def get_time_perf_counter(sync: bool = False) -> float:
    if sync:
        torch.cuda.synchronize()
    return time.perf_counter()


def secs_to_millis(secs: float) -> int:
    """
    Converts seconds to milliseconds.
    """
    return round(secs * 1e3)


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
