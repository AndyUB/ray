import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import open_clip
from open_clip.transformer import Transformer, VisionTransformer


# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="FSDP CLIP Training")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="batch size per GPU"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local rank for distributed training"
    )
    parser.add_argument(
        "--data-path", type=str, default="./data", help="path to dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output", help="path to save outputs"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    return parser.parse_args()


# Set up mixed precision policy for FSDP
def get_mixed_precision_policy():
    bfloat16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    return bfloat16_policy


# Initialize process group
def init_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


# Setup FSDP
def setup_fsdp(model):
    # Auto wrapping policy to shard transformer blocks
    transformer_cls = (Transformer, VisionTransformer)
    wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls=transformer_cls,
    )

    # Set up FSDP configuration
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=get_mixed_precision_policy(),
        device_id=torch.cuda.current_device(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        cpu_offload=CPUOffload(offload_params=False),
    )

    return model


# Load CLIP model from open_clip
def load_clip_model():
    model, _, _ = open_clip.create_model_and_transforms(
        model_name="ViT-L-14",
        pretrained=None,  # Set to a checkpoint name if you want to start from a pretrained model
    )
    return model


# CLIP Loss function
class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image_features, text_features):
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # Create labels (diagonal of ones matrix)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        # Symmetric loss function
        loss_i = nn.functional.cross_entropy(logits_per_image, labels)
        loss_t = nn.functional.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        return loss


# Simplified dataset loader - replace with your actual dataset
def get_dataloader(args, is_distributed=True):
    # This is a placeholder - replace with your actual dataset
    # For example:
    # from your_dataset import YourCLIPDataset
    # dataset = YourCLIPDataset(args.data_path)

    # For demonstration purposes:
    from torch.utils.data import Dataset

    class DummyCLIPDataset(Dataset):
        def __init__(self, size=10000):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Simulate an image and text pair
            image = torch.randn(3, 224, 224)
            text = torch.randint(
                0, 49408, (77,)
            )  # 49408 is the vocab size, 77 is the context length
            return image, text

    dataset = DummyCLIPDataset()

    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    return dataloader


# Training function
def train(args):
    # Initialize distributed environment
    init_distributed()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)

    # Create model and move to GPU
    model = load_clip_model()

    # Wrap model with FSDP
    model = setup_fsdp(model)

    # Set up optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Set up loss function
    criterion = CLIPLoss()

    # Get data loader
    train_loader = get_dataloader(args, is_distributed=True)

    # Training loop
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()

        for i, (images, texts) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            texts = texts.cuda(non_blocking=True)

            # Forward pass
            image_features, text_features = model(images, texts)

            # Compute loss
            loss = criterion(image_features, text_features)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log progress
            if i % 10 == 0 and dist.get_rank() == 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

        # Save checkpoint at the end of each epoch
        if dist.get_rank() == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = os.path.join(
                args.output_dir, f"clip_checkpoint_epoch_{epoch}.pt"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                save_path,
            )
            print(f"Checkpoint saved to {save_path}")


# Main function to launch training
def main():
    args = parse_args()

    # Set environment variables for distributed training
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # Launch with 4 GPUs
    world_size = 4
    mp.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )


def main_worker(rank, world_size, args):
    # Set up distributed environment
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    # Start training
    train(args)


if __name__ == "__main__":
    main()
