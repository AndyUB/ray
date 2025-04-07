import logging
import os
import sys
from collections import defaultdict
from typing import Any, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from src.core.common import get_timing_event_torch, millis_to_micros
from src.core.llama3.model import LLAMA_8B as LLAMA
from src.core.llama3.model import TransformerWrapped
from torch.utils.data import DataLoader, TensorDataset


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d -- %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
logger.info("Welcome to Downton Abbey!")


def run_torch_fsdp(args: Dict[str, Any]) -> None:
    logger = logging.getLogger(__name__)
    set_seed(998244353)

    # Initialize accelerator
    accelerator = Accelerator()

    device = accelerator.device
    model_args = LLAMA
    logger.info(f"model_args: {model_args}")

    # Create model
    model = TransformerWrapped(model_args).half().to(device)

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
            device=device,
        )
        target_ids = torch.randn(
            batch_size,
            seq_len,
            model_args.vocab_size,
            device=device,
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
    train_dataloader = DataLoader(train_dataset, shuffle=True)

    # Set up the criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
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

        if accelerator.is_main_process:
            logger.info(f"iter: {i}")

        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        torch.cuda.synchronize()
        start = get_timing_event_torch()

        # Forward pass
        fw_start = get_timing_event_torch()
        pred = model(input_ids)
        fw_end = get_timing_event_torch()

        # Loss computation
        bw_loss_comp_start = get_timing_event_torch()
        loss = criterion(pred, target_ids)
        bw_loss_comp_end = get_timing_event_torch()

        # Backward pass
        bw_grad_start = get_timing_event_torch()
        accelerator.backward(loss)
        bw_grad_end = get_timing_event_torch()

        # Update weights
        others_upd_start = get_timing_event_torch()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        others_upd_end = get_timing_event_torch()

        torch.cuda.synchronize()
        barrier_start = get_timing_event_torch()
        accelerator.wait_for_everyone()
        end = get_timing_event_torch()

        torch.cuda.synchronize()
        total_ms = start.elapsed_time(end)

        def log(key: str, elapse_ms: float):
            elapse_us = millis_to_micros(elapse_ms)
            elapses[key].append(elapse_us)
            if accelerator.is_main_process:
                logger.warning(
                    f"rank: {accelerator.process_index}, {key} elapse: {elapse_us} us, percent: {round(elapse_ms / total_ms * 100, 1)}%"
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
            if accelerator.is_main_process:
                logger.warning("")


if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    args = {
        "batch_size": 1,
        "seq_len": 1024,
        "num_iters": 20,
    }
    run_torch_fsdp(args)
