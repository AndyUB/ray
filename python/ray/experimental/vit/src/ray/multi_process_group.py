from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import List, Dict, Any

import ray
from ray.air._internal.util import find_free_port
from ray.air.util.torch_dist import _init_torch_distributed


@dataclass
class TorchDistributedConfig:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    master_addr: str
    master_port: str
    gpu_ids: List[int]


class BaseWorker:
    def __init__(self) -> None:
        pass

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "gpu_ids": ray.get_gpu_ids(),
            "address": ray.util.get_node_ip_address(),
            "port": find_free_port(),
        }

    def init_dist_group(self, dist_config: TorchDistributedConfig) -> None:
        self.dist_config = dist_config
        _init_torch_distributed(
            init_method="env", backend="nccl", **asdict(dist_config)
        )
        print(f"Rank {self.dist_config.rank}: Initialized")
        if self.dist_config.rank == 0:
            print(asdict(self.dist_config))


def initialize_dist_group(workers: List["ray.actor.ActorHandle"]) -> None:
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

            rank += 1
            local_rank += 1

            dist_configs[worker_id] = worker_config

    ray.get(
        [
            worker.init_dist_group.remote(dist_configs[worker_id])
            for worker_id, worker in enumerate(workers)
        ]
    )
    print("Finished initializing distributed process group.")


if __name__ == "__main__":
    workers = [BaseWorker.remote() for _ in range(2)]
    initialize_dist_group(workers)

    workers = [BaseWorker.remote() for _ in range(6)]
    initialize_dist_group(workers)
