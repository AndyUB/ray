import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

import ray

logger = logging.getLogger(__name__)


@dataclass
class TorchDistributedConfig:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    master_addr: str
    master_port: str
    gpu_ids: List[int]


def init_torch_distributed(
    workers,
    visible_gpus: Optional[List[int]] = None,  # [HACK]
):
    """Initialize PyTorch Distributed Process Group for a set of workers."""
    worker_metadata = ray.get([worker.get_metadata.remote() for worker in workers])

    for worker_id, metadata in enumerate(worker_metadata):
        metadata["worker_id"] = worker_id

    aggregated_metadata = defaultdict(list)

    for metadata in worker_metadata:
        aggregated_metadata[metadata["address"]].append(metadata)

    for metadata_list_per_ip in aggregated_metadata.values():
        metadata_list_per_ip.sort(key=lambda x: x["gpu_ids"])

    rank = 0
    world_size = len(workers)
    dist_configs = dict()

    for metadata_list_per_ip in aggregated_metadata.values():
        local_rank = 0
        local_world_size = len(metadata_list_per_ip)
        visible_device_ids = []

        for metadata in metadata_list_per_ip:
            visible_device_ids += metadata["gpu_ids"]

        # [HACK]
        if visible_gpus is not None:
            visible_device_ids = visible_gpus

        for metadata in metadata_list_per_ip:
            if rank == 0:
                master_addr = metadata["address"]
                master_port = metadata["port"]

            worker_id = metadata["worker_id"]
            worker_config = TorchDistributedConfig(
                rank=rank,
                local_rank=local_rank,
                world_size=world_size,
                local_world_size=local_world_size,
                master_addr=master_addr,
                master_port=master_port,
                gpu_ids=visible_device_ids,
            )
            print(worker_config)

            rank += 1
            local_rank += 1

            dist_configs[worker_id] = worker_config

    ray.get(
        [
            worker.init_torch_distributed.remote(dist_configs[worker_id])
            for worker_id, worker in enumerate(workers)
        ]
    )
    logger.info("Finished initializing distributed process group.")
