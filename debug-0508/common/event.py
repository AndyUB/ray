import torch


def get_torch_cuda_event() -> torch.cuda.Event:
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def get_elapsed_us(ev1: torch.cuda.Event, ev2: torch.cuda.Event) -> float:
    return ev1.elapsed_time(ev2) * 1e3
