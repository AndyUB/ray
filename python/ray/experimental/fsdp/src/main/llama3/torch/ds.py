import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import torch
import deepspeed
from deepspeed import comm

from src.core.common import get_timing_event_torch, millis_to_micros
from src.core.llama3.model import LLAMA_3B as LLAMA
from src.core.llama3.model import TransformerWrapped
from torch.utils.data import TensorDataset


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")

deepspeed_config_dict = {
    "train_batch_size": 2,  # Will be overwritten by training loop logic
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 1,
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "none"},
        "offload_param": {"device": "none"},
        "stage3_param_persistence_threshold": 1e5,  # Optional: can help perf
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
    },
    "bf16": {"enabled": True},
    "fp16": {"enabled": False},  # Explicitly disabled since using bf16
    "zero_allow_untested_optimizer": True,  # Required for non-standard opt
    "wall_clock_breakdown": False,  # Optional, useful for profiling
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_torch_fsdp(args: Dict[str, Any]) -> None:
    logger = logging.getLogger(__name__)
    set_seed(998244353)

    model_args = LLAMA
    logger.info(f"model_args: {model_args}")

    # Create model
    model = TransformerWrapped(model_args).half()

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    # Create a simple scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args["num_iters"], eta_min=1e-7
    )

    # Create random data for training
    batch_size = args["batch_size"]
    seq_len = args["seq_len"]

    # Creating random dataset
    def get_random_batch():
        input_ids = torch.randint(
            0,
            model_args.vocab_size,
            (batch_size, seq_len),
        )
        target_ids = torch.randn(
            batch_size,
            seq_len,
            model_args.vocab_size,
        )
        return input_ids, target_ids

    # Create a small dataset of random tensors
    inputs = []
    targets = []
    for _ in range(1):
        input_batch, target_batch = get_random_batch()
        inputs.append(input_batch)
        targets.append(target_batch)

    train_dataset = TensorDataset(torch.cat(inputs), torch.cat(targets))
    # train_dataloader = DataLoader(train_dataset, shuffle=True)

    # Set up the criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Prepare everything with accelerator
    model_engine, optimizer, train_dataloader, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        lr_scheduler=scheduler,
        config=deepspeed_config_dict,
    )
    device = model_engine.device

    size_bytes = sum(
        p.numel() * p.element_size() for p in model_engine.module.parameters()
    )
    logger.warning(f"Model size: {round(size_bytes / 1024 / 1024)} MiB")

    # For tracking time
    elapses = defaultdict(list)

    # Training loop
    for i in range(args["num_iters"]):
        # Get batch from dataloader (it will cycle through if needed)
        try:
            input_ids, target_ids = next(train_iter)
        except Exception:
            train_iter = iter(train_dataloader)
            input_ids, target_ids = next(train_iter)

        if comm.get_rank() == 0:
            logger.info(f"iter: {i}")

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        torch.cuda.synchronize()
        start = get_timing_event_torch()

        # Forward pass
        fw_start = get_timing_event_torch()
        pred = model_engine(input_ids)
        fw_end = get_timing_event_torch()

        # Loss computation
        bw_loss_comp_start = get_timing_event_torch()
        loss = criterion(pred, target_ids)
        bw_loss_comp_end = get_timing_event_torch()

        # Backward pass
        bw_grad_start = get_timing_event_torch()
        model_engine.backward(loss)
        bw_grad_end = get_timing_event_torch()

        # Update weights
        others_upd_start = get_timing_event_torch()
        model_engine.step()
        others_upd_end = get_timing_event_torch()

        torch.cuda.synchronize()
        barrier_start = get_timing_event_torch()
        comm.barrier()
        end = get_timing_event_torch()

        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)

        def log(key: str, elapse_ms: float):
            elapse_us = millis_to_micros(elapse_ms)
            elapses[key].append(elapse_us)
            if comm.get_rank() == 0:
                logger.warning(
                    f"rank: {comm.get_rank()}, {key} elapse: {elapse_us} us, percent: {round(elapse_ms / total_ms * 100, 1)}%"
                )

        if i > 0:
            log("total", total_ms)
            log("actor.total", total_ms)
            log("fw.total", fw_start.elapsed_time(fw_end))
            log("bw.total", bw_loss_comp_start.elapsed_time(bw_grad_end))
            log("bw.loss.comp", bw_loss_comp_start.elapsed_time(bw_loss_comp_end))
            log("bw.grad", bw_grad_start.elapsed_time(bw_grad_end))
            log("others.upd", others_upd_start.elapsed_time(others_upd_end))
            log("barrier", barrier_start.elapsed_time(end))
            if comm.get_rank() == 0:
                logger.warning("")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
    args = {
        "batch_size": 1,
        "seq_len": 1024,
        "num_iters": 20,
    }
    run_torch_fsdp(args)
