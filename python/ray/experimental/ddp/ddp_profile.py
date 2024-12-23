import logging
import os
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity

import ray
from ray.air._internal import torch_utils
from ray.dag import InputNode, MultiOutputNode
from ray.experimental.collective import allreduce


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s %(filename)s:%(lineno)d %(funcName)s] %(message)s",
)

SECOND_TO_MICROSECOND_RATIO = 1e6


def secs_to_micros(secs: float) -> int:
    """
    Converts seconds to microseconds (rounded).
    """
    return round(secs * SECOND_TO_MICROSECOND_RATIO)


@dataclass
class Config:
    """Configuration for the demo DDP model."""

    # Model config.
    num_layers: int
    # The layer is a square (n * n).
    layer_size: int
    dtype: torch.dtype

    # Training config.
    num_iters: int
    learning_rate: float
    num_actors: int

    # Check correctness flag.
    check_correctness: bool
    # Output file.
    output_file: str
    # Breakdown performance flag.
    breakdown_performance: bool


# Set RNG seed for deterministic results.
SEED = 42


class LayeredModel(torch.nn.Module):
    """
    A model that is a chain of (linear, relu) layers.

    Args:
        layer_size: Size of each layer. Each layer is a square (n * n).
        num_layers: Number of layers in the model.
        device: Device the model is on.
        dtype: Data type of the parameters in the model.
        lr: Learning rate for the optimizer.
    """

    def __init__(
        self,
        layer_size: int,
        num_layers: int,
        device: torch.device,
        dtype: torch.dtype,
        lr: float,
    ):
        torch.manual_seed(SEED)

        super(LayeredModel, self).__init__()

        self.layers: List[torch.nn.Module] = []
        for _ in range(num_layers):
            # For simplicity, no bias.
            self.layers.append(
                torch.nn.Linear(
                    layer_size, layer_size, device=device, dtype=dtype, bias=False
                )
            )
            self.layers.append(torch.nn.ReLU())
        self.layers: nn.ModuleList = nn.ModuleList(self.layers)
        self.inputs: List[torch.Tensor] = []
        self.activations: List[torch.Tensor] = []
        self.lr: float = lr
        self.criterion = nn.MSELoss()
        # [TODO] Use a single optimizer for all layers.
        # [TODO] performance comparison: 1 optim vs many?
        self.optimizers: List[optim.SGD] = [
            optim.SGD(self.layers[2 * i].parameters(), lr=self.lr)
            for i in range(num_layers)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """
        Forward pass for a single layer. Cache the input and activation.

        Args:
            x: Input for this layer.
            layer_idx: Index of the layer.
        """
        self.inputs.append(x)
        linear_layer: torch.nn.Linear = self.layers[2 * layer_idx]
        y: torch.Tensor = linear_layer(x)
        relu_activation: torch.nn.Module = self.layers[2 * layer_idx + 1]
        z: torch.Tensor = relu_activation(y)
        self.activations.append(z)
        return z

    def backward_layer(
        self, grad: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward pass for a single layer. Return the gradient of the loss with
        respect to the input and that with respect to the weight.
        """
        z: torch.Tensor = self.activations[layer_idx]
        x: torch.Tensor = self.inputs[layer_idx]
        layer: torch.nn.Linear = self.layers[2 * layer_idx]
        W: torch.Tensor = layer.weight
        # Because the backward pass is done layer by layer, it is necessary to
        # retain the graph unless this is the first layer. Otherwise, the graph
        # is freed after use and cannot be backpropagated through a second time.
        retain_graph = layer_idx != 0
        z.backward(gradient=grad, retain_graph=retain_graph, inputs=[W, x])
        return x.grad, W.grad

    def update_layer(
        self, grad: torch.Tensor, layer_idx: int, check_correctness: bool
    ) -> Optional[torch.Tensor]:
        """
        Update the specified layer with the given gradient. Return the layer
        weight if check correctness, otherwise return None.

        Args:
            grad: Gradient for layer update.
            layer_idx: Index of the layer.
            check_correctness: Whether correctness is checked.
        """
        layer: torch.nn.Linear = self.layers[2 * layer_idx]
        [param for param in layer.parameters()][0].grad = grad
        optimizer = self.optimizers[layer_idx]
        optimizer.step()
        if check_correctness:
            return layer.weight
        else:
            return None


@ray.remote
class RayDDPWorker:
    """
    An actor class wrapper around the pytorch model.

    Args:
        num_layers: Number of layers in the model.
        layer_size: Size of each layer. Each layer is a square (n * n).
        world_size: Number of actors.
        dtype: Data type of the parameters in the model.
        lr: Learning rate for the optimizer.
        check_correctness: Whether correctness is checked.
        breakdown_performance: Whether to print performance breakdown.
    """

    def __init__(
        self,
        num_layers: int,
        layer_size: int,
        world_size: int,
        dtype: torch.dtype,
        lr: float,
        check_correctness: bool,
        breakdown_performance: bool,
    ):
        self.num_layers = num_layers
        self.layer_size = layer_size
        # Each device has a single GPU.
        self.device = torch_utils.get_devices()[0]

        self.model: LayeredModel = LayeredModel(
            layer_size, num_layers, self.device, dtype, lr
        )
        self.world_size: int = world_size

        self.check_correctness = check_correctness
        # [TODO] remove manual timing and use profiler
        self.breakdown_performance = breakdown_performance
        self.it = 0
        self.start_time: float = None
        self.pre_forward_time: float = None
        self.forward_times: List[Tuple[float, Tuple]] = None
        self.loss_time: float = None
        self.pre_backward_time: float = None
        self.backward_times: List[Tuple[float, float]] = None
        self.update_times: List[Tuple[float, float]] = None
        self.end_time: float = None

        self.x: torch.Tensor = None
        self.y: torch.Tensor = None

        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        )
        self.profiler.__enter__()

    def tensor_to_device(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Move the input and ground truth to the device. The input and ground truth
        were on the CPU so they need to be moved.

        Args:
            x: Input.
            y: Ground truth.
        """
        self.x = x.to(self.device)
        self.y = y.to(self.device)

    def start_train(self) -> None:
        """
        Start the training process for one iteration. Clear the old gradients,
        stored inputs, and activations from last iteration.
        """
        if self.breakdown_performance:
            self.start_time = time.perf_counter()

        self.model.zero_grad()
        self.model.inputs = []
        self.model.activations = []

        if self.breakdown_performance:
            self.forward_times = []
            self.backward_times = []
            self.update_times = []

    def forward(self, placeholder: Any) -> torch.Tensor:
        """
        Forward pass for the model.
        1. Compute the prediction with the input.
        2. Compute the loss from the prediction and the ground truth.
        3. Compute and return the gradient of the loss with respect to the prediction.

        Args:
            placeholder: Unused placeholder for the input.

        Returns:
            Gradient of the loss with respect to the prediction.
        """
        self.start_train()
        x = self.x
        y = self.y
        if self.breakdown_performance:
            self.pre_forward_time = time.perf_counter()

        for i in range(self.num_layers):
            forward_start_time = None
            if self.breakdown_performance:
                forward_start_time = time.perf_counter()
            x = self.model.forward_layer(x, i)
            forward_end_time = None
            if self.breakdown_performance:
                forward_end_time = time.perf_counter()
                self.forward_times.append((forward_start_time, forward_end_time))

        pred = x
        loss: torch.Tensor = self.model.criterion(pred, y)
        if self.breakdown_performance:
            self.loss_time = time.perf_counter()

        # Compute the gradient of the loss with respect to the prediction.
        # Retain the graph (i.e., not free the graph) for subsequent backprop
        # computations.
        loss.backward(retain_graph=True, inputs=[pred])
        if self.breakdown_performance:
            self.pre_backward_time = time.perf_counter()
        return pred.grad, None

    def backward(
        self, layer_idx: int, grad: Tuple[torch.Tensor, Optional[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the backward pass for the specified layer.

        Args:
            layer_idx: Index of the layer.
            grad: `grad[0]` is the gradient of the loss with respect to this
                layer's output. This is a workaround for the issue:
                https://github.com/ray-project/ray/issues/48522

        Returns:
            Tuple of the gradients of the loss with respect to the input and the
            weight of this layer.
        """
        backward_start_time = None
        if self.breakdown_performance:
            backward_start_time = time.perf_counter()
        # No need to move the gradient because it is already on this device.
        bp_grad, _ = grad
        result = self.model.backward_layer(bp_grad, layer_idx)
        backward_end_time = None
        if self.breakdown_performance:
            backward_end_time = time.perf_counter()
            self.backward_times.append((backward_start_time, backward_end_time))
        return result

    def update(self, layer_idx: int, grad: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Update the weights of the specified layer with the given allreduced gradient.

        Args:
            layer_idx: Index of the layer.
            grad: Allreduced gradient for this layer.

        Returns:
            The updated weights of this layer if correctness is checked, otherwise
            None.
        """
        update_start_time = None
        if self.breakdown_performance:
            update_start_time = time.perf_counter()
        # No need to move the gradient because it is already on this device.
        # For mathematical equivalence, divide the allreduced gradient by the
        # world size (i.e., the number of actors).
        grad /= self.world_size
        result = self.model.update_layer(grad, layer_idx, self.check_correctness)
        update_end_time = None
        if self.breakdown_performance:
            update_end_time = time.perf_counter()
            self.update_times.append((update_start_time, update_end_time))
        return result

    def finish_train(
        self, *updates: Optional[torch.Tensor]
    ) -> List[Optional[torch.Tensor]]:
        """
        Finish the current iteration of training by gather all results from weight
        updates across different layers.

        Args:
            updates: A tuple of any number of results from weight updates.
                If correctness is checked, an update result is the updated weight of
                a layer. Otherwise, it is None.

        Returns:
            The list of all results from weight updates.
        """
        if self.breakdown_performance:
            self.end_time = time.perf_counter()

            print(f"start time: {self.start_time}")
            print(f"pre forward time: {self.pre_forward_time}")
            for i, (start, end) in enumerate(self.forward_times):
                print(f"forward time layer {i}: start: {start}, end: {end}")
            print(f"loss time: {self.loss_time}")
            print(f"pre backward time: {self.pre_backward_time}")
            for i, (start, end) in enumerate(self.backward_times):
                print(f"backward time layer {i}: start: {start}, end: {end}")
            for i, (start, end) in enumerate(self.update_times):
                print(f"update time layer {i}: start: {start}, end: {end}")
            print(f"end time: {self.end_time}")
            print()

            self.start_time = secs_to_micros(self.start_time)
            self.pre_forward_time = secs_to_micros(self.pre_forward_time)
            for i, (start, end) in enumerate(self.forward_times):
                self.forward_times[i] = (secs_to_micros(start), secs_to_micros(end))
            self.loss_time = secs_to_micros(self.loss_time)
            self.pre_backward_time = secs_to_micros(self.pre_backward_time)
            for i, (start, end) in enumerate(self.backward_times):
                self.backward_times[i] = (secs_to_micros(start), secs_to_micros(end))
            for i, (start, end) in enumerate(self.update_times):
                self.update_times[i] = (secs_to_micros(start), secs_to_micros(end))
            self.end_time = secs_to_micros(self.end_time)

            print(f"=============== Iteration {self.it} Elapses =================")
            print(f"pre forward elapse: {self.pre_forward_time - self.start_time}")
            for i, (start, end) in enumerate(self.forward_times):
                print(f"forward layer {i} elapse: {end - start}")
            print(f"loss elapse: {self.loss_time - self.forward_times[-1][1]}")
            print(f"pre backward elapse: {self.pre_backward_time - self.loss_time}")
            for i, (start, end) in enumerate(self.backward_times):
                print(f"backward layer {i} elapse: {end - start}")
            for i, (start, end) in enumerate(self.update_times):
                print(
                    f"allreduce layer {i} elapse: {start - self.backward_times[i][1]}"
                )
                print(f"update layer {i} elapse: {end - start}")
            print(f"total elapse: {self.end_time - self.start_time}")
            print_time = secs_to_micros(time.perf_counter())
            print(f"print elapse: {print_time - self.end_time}")
            print(
                "====================================================================="
            )
            print()

            self.it += 1

        return updates if self.check_correctness else None

    def get_grad_to_reduce(
        self, grad: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Extracts the gradient to be reduced.

        Args:
            grad: `grad[1]` is the gradient of the loss with respect to the weight.
                This gradient is used in the allreduce operation.

        Returns:
            The gradient to be reduced.

        When an allreduce binds a class method output with `num_returns > 1`,
        an error is thrown. This is a workaround.
        See: https://github.com/ray-project/ray/issues/48522
        """
        _, reduce_grad = grad
        return reduce_grad

    def stop_profiler(self, rank) -> None:
        """
        Stop the profiler and save the trace.

        rank: Rank of the actor.
        """
        self.profiler.__exit__(None, None, None)
        self.profiler.export_chrome_trace(
            f"ray_ddp_profiling_{self.num_layers}_layers_"
            f"{self.layer_size}_size_{rank}_rank.json"
        )
        # The profiler cannot be reused after it is stopped.
        self.profiler = None


def generate_input_output(config: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate input `x` and output `y` for training.

    Args:
        config: Model and training configurations.

    Returns:
        Input `x` and ground truth `y`.
    """
    layer_size = config.layer_size
    num_actors = config.num_actors
    dtype = config.dtype

    shape = (num_actors * layer_size, layer_size)
    numel = shape[0] * shape[1]

    x = torch.arange(numel, dtype=dtype, requires_grad=True) / numel
    x = x.reshape(shape)
    y = torch.arange(numel, dtype=dtype) / numel
    y = y.reshape(shape)

    return x, y


def run_torch(config: Config) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    """
    Run PyTorch without DDP.

    Args:
        config: Model and training configurations.

    Returns:
        Weights of all layers after each iteration if correctness is checked, and
        the average elapse across all iterations.
    """

    # To ensure that the model parameters are initialized in the same way across
    # different training methods (PyTorch without DDP, PyTorch with DDP, and Ray
    # with DDP), the model must be initialized on GPU.
    device = "cuda:0"
    model = LayeredModel(
        config.layer_size,
        config.num_layers,
        device,
        config.dtype,
        config.learning_rate,
    )
    criterion = model.criterion
    optimizer = optim.SGD(model.parameters(), lr=model.lr)

    weights: Optional[List[List[torch.Tensor]]] = None
    if config.check_correctness:
        weights = []
    elapses: List[float] = []
    for i in range(config.num_iters):
        x, y = generate_input_output(config)
        x = x.to(device)
        y = y.to(device)
        start = time.perf_counter()
        optimizer.zero_grad()
        pred: torch.Tensor = model(x)
        loss: torch.Tensor = criterion(pred, y)
        loss.backward()
        optimizer.step()
        end = time.perf_counter()

        if config.check_correctness:
            cur_iter_weights: List[torch.Tensor] = []
            for i in range(0, len(model.layers), 2):
                layer: torch.nn.Linear = model.layers[i]
                cur_iter_weights.append(torch.clone(layer.weight))
            weights.append(cur_iter_weights)

        elapse = end - start
        elapses.append(elapse)

    avg_elapse = print_elapses(elapses, "torch")
    return weights, avg_elapse


def run_torch_ddp(config: Config) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    """
    Run DDP with PyTorch.

    Args:
        config: Model and training configurations.

    Returns:
        Weights of all layers after each iteration if correctness is checked,
        and the average elapse across all iterations.
    """
    n_gpus = torch.cuda.device_count()
    assert (
        n_gpus >= config.num_actors
    ), f"Requires at least {config.num_actors} GPUs to run, but got {n_gpus}"
    world_size = config.num_actors

    mp.set_start_method("spawn", force=True)
    # Use a multiprocessing manager to share data across devices.
    with mp.Manager() as manager:
        weights_dict = None
        if config.check_correctness:
            weights_dict = manager.dict()
        elapses_dict = manager.dict()
        # [TODO] is spawn the only way for multiprocessing?
        mp.spawn(
            run_torch_ddp_per_process,
            args=(world_size, weights_dict, elapses_dict, config),
            nprocs=world_size,
            join=True,
        )
        weights = None
        if config.check_correctness:
            weights = get_torch_ddp_weights_per_device(weights_dict, world_size)
        avg_elapse = get_torch_ddp_e2e_elapse(elapses_dict)
        return weights, avg_elapse


def run_torch_ddp_per_process(
    rank: int,
    world_size: int,
    weights_dict: Optional[Dict[int, List[List[torch.Tensor]]]],
    elapses_dict: Dict[int, int],
    config: Config,
) -> None:
    """
    Run DDP with PyTorch in one process.

    Args:
        rank: The rank of this process.
        world_size: The total number of processes.
        weights_dict: A dictionary to store all weights of this process.
        elapses_dict: A dictionary to store the average elapse of this process.
        config: Configurations. If config.check_correctness, store the weights
            into weights_dict.
    """

    if config.check_correctness:
        assert weights_dict is not None

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    # Initialize the process group.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Create model on GPU with id rank.
    model = LayeredModel(
        config.layer_size,
        config.num_layers,
        f"cuda:{rank}",
        config.dtype,
        config.learning_rate,
    )
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = model.criterion
    optimizer = optim.SGD(ddp_model.parameters(), lr=model.lr)

    num_actors = config.num_actors

    weights: Optional[List[List[torch.Tensor]]] = None
    if config.check_correctness:
        weights = []
    elapses = []
    timestamps = []
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for i in range(config.num_iters):
            x, y = generate_input_output(config)
            x = torch.tensor_split(x, num_actors)[rank].to(rank)
            y = torch.tensor_split(y, num_actors)[rank].to(rank)
            start = time.perf_counter()
            optimizer.zero_grad()
            pre_forward = time.perf_counter()
            prediction: torch.Tensor = ddp_model(x)
            post_forward = time.perf_counter()
            loss: torch.Tensor = loss_fn(prediction, y)
            pre_backward = time.perf_counter()
            loss.backward()
            post_backward = time.perf_counter()
            optimizer.step()
            end = time.perf_counter()
            timestamps.append(
                (start, pre_forward, post_forward, pre_backward, post_backward, end)
            )

            if config.check_correctness:
                cur_iter_weights = []
                for i in range(0, len(model.layers), 2):
                    layer: torch.nn.Linear = model.layers[i]
                    cur_iter_weights.append(torch.clone(layer.weight))
                weights.append(cur_iter_weights)

            elapse = end - start
            elapses.append(elapse)
            prof.step()

    prof.export_chrome_trace(
        f"torch_ddp_profiling_{config.num_layers}_layers_"
        f"{config.layer_size}_size_{rank}_rank.json"
    )
    for i, (
        start,
        pre_forward,
        post_forward,
        pre_backward,
        post_backward,
        end,
    ) in enumerate(timestamps):
        logger.info(f"===========Torch DDP Iteration {i}===========")
        logger.info(
            f"start: {secs_to_micros(start)}, "
            f"pre_forward: {secs_to_micros(pre_forward)}, "
            f"post_forward: {secs_to_micros(post_forward)}, "
            f"pre_backward: {secs_to_micros(pre_backward)}, "
            f"post_backward: {secs_to_micros(post_backward)}, "
            f"end: {secs_to_micros(end)}"
        )
        zero_grad_elapse = pre_forward - start
        forward_elapse = post_forward - pre_forward
        loss_elapse = pre_backward - post_forward
        backward_elapse = post_backward - pre_backward
        update_elapse = end - post_backward
        logger.info(
            f"zero_grad: {secs_to_micros(zero_grad_elapse)}, "
            f"forward: {secs_to_micros(forward_elapse)}, "
            f"loss: {secs_to_micros(loss_elapse)}, "
            f"backward: {secs_to_micros(backward_elapse)}, "
            f"update: {secs_to_micros(update_elapse)}"
        )
    avg_elapse = print_elapses(elapses, f"torch ddp rank: {rank}", rank)

    # Destroy the process group.
    dist.destroy_process_group()

    def detach_all(
        weights_across_iters: List[List[torch.Tensor]],
    ) -> List[List[torch.Tensor]]:
        """
        Detach all tensors in order to pass tensors across devices.
        If a tensor is not detached, serialization will fail.

        Args:
            weights_across_iters: Weights of all layers across all iterations.

        Returns:
            Detached weights of all layers across all iterations.
        """
        all_iters_detached: List[List[torch.Tensor]] = []
        for single_iter_tensors in weights_across_iters:
            detached: List[torch.Tensor] = []
            for tensor in single_iter_tensors:
                detached.append(tensor.detach().cpu())
            all_iters_detached.append(detached)
        return all_iters_detached

    if config.check_correctness:
        weights_dict[rank] = detach_all(weights)
    elapses_dict[rank] = avg_elapse


def get_torch_ddp_weights_per_device(
    weights_dict: Dict[int, List[List[torch.Tensor]]], world_size: int
) -> List[List[torch.Tensor]]:
    """
    Extract the per-device weights of all layers after each iteration.
    Check that the model is consistent across all devices after each iteration.

    Args:
        weights_dict: Dictionary that maps ranks (device ids) to its weights of
            all layers after each iteration.
        world_size: The total number of devices.

    Returns:
        Per-device weights of all layers after each iteration.
    """
    weights_across_devices = list(dict(weights_dict).values())
    assert len(weights_across_devices) == world_size
    weights_per_device = weights_across_devices[0]
    # Weights on each device.
    for i in range(1, world_size):
        cur_device_weights = weights_across_devices[i]
        assert len(weights_per_device) == len(cur_device_weights)
        # Weights per iteration on each device.
        for j in range(len(weights_per_device)):
            assert len(weights_per_device[j]) == len(cur_device_weights[j])
            # Weights per layer in each iteration.
            for k in range(len(weights_per_device[j])):
                assert torch.equal(weights_per_device[j][k], cur_device_weights[j][k])
    return weights_per_device


def get_torch_ddp_e2e_elapse(elapses_dict: Dict[int, int]) -> int:
    """
    Get the end-to-end elapse for all devices. The maximum elapse across all
    devices is used to approximate the end-to-end elapse.

    Args:
        elapses_dict: Dictionary that maps ranks (device ids) to its average
            elapse across iterations.

    Returns:
        The approximate end-to-end elapse for all devices.
    """
    return max(elapses_dict.values())


def run_ray_ddp(config: Config) -> Tuple[Optional[List[List[torch.Tensor]]], int]:
    """
    Run DDP using compiled graph and allreduce in Ray.

    Args:
        config: Model and training configurations.

    Returns:
        Per-device weights of all layers after each iteration if correctness is checked,
        and the average end-to-end elapse.
    """
    ray.init()
    if sum(node["Resources"].get("GPU", 0) for node in ray.nodes()) < config.num_actors:
        raise ValueError(f"Needs at least {config.num_actors} GPUs")

    actor_cls = RayDDPWorker.options(num_gpus=1)
    num_layers, layer_size = config.num_layers, config.layer_size
    num_actors = config.num_actors
    actors = [
        actor_cls.remote(
            num_layers,
            layer_size,
            num_actors,
            config.dtype,
            config.learning_rate,
            config.check_correctness,
            config.breakdown_performance,
        )
        for _ in range(num_actors)
    ]

    with InputNode() as inp:
        grads = [actor.forward.bind(inp) for actor in actors]
        output = []
        for j in reversed(range(num_layers)):
            for i, actor in enumerate(actors):
                grads[i] = actor.backward.bind(j, grads[i])
            reduced_grads = allreduce.bind(
                [
                    actor.get_grad_to_reduce.bind(grads[i])
                    for i, actor in enumerate(actors)
                ]
            )
            updates = [
                actor.update.bind(j, reduced_grad)
                for actor, reduced_grad in zip(actors, reduced_grads)
            ]
            output.append(updates)
        ends = [
            actor.finish_train.bind(
                *[output[j][i] for j in reversed(range(num_layers))]
            )
            for i, actor in enumerate(actors)
        ]
        dag = MultiOutputNode(ends)

    compiled_dag = dag.experimental_compile()

    x, y = generate_input_output(config)
    xs = torch.tensor_split(x, num_actors)
    ys = torch.tensor_split(y, num_actors)
    move_tensor_refs = [
        actor.tensor_to_device.remote(xs[i], ys[i]) for i, actor in enumerate(actors)
    ]
    ray.get(move_tensor_refs)

    weights = None
    if config.check_correctness:
        weights = []
    elapses = []
    for i in range(config.num_iters):
        start = time.perf_counter()
        # Use None as a placeholder.
        ref = compiled_dag.execute(None)
        # [TODO] print timestamp before ray.get
        # [TODO] use mature profiler
        # If correctness is not checked, the result is None.
        cur_iter_weights = ray.get(ref)
        end = time.perf_counter()
        if config.check_correctness:
            weights.append(cur_iter_weights)
        elapse = end - start
        elapses.append(elapse)

    ray.get([actor.stop_profiler.remote(rank) for rank, actor in enumerate(actors)])
    avg_elapse = print_elapses(elapses, "ray ddp")
    compiled_dag.teardown()

    for actor in actors:
        ray.kill(actor)
    ray.shutdown()

    if config.check_correctness:
        weights = get_ray_ddp_weights_per_device(weights, config.num_actors)
    return weights, avg_elapse


def get_ray_ddp_weights_per_device(
    weights: List[List[List[torch.Tensor]]], num_actors: int
) -> List[List[torch.Tensor]]:
    """
    Check that the model is consistent across all ranks after each iteration
    of training. Deduplicate the weights after the checks.

    Args:
        weights: Weights from running Ray DDP for a number of iterations.
            `weights[i]` is the weights across all actors for the ith iteration.

    Returns:
        Per-device weights of all layers after each iteration.
    """
    weights_per_device: List[List[torch.Tensor]] = []
    for single_iter_weights in weights:
        per_device_len = len(single_iter_weights[0])
        for i in range(1, num_actors):
            for j in range(per_device_len):
                assert torch.equal(
                    single_iter_weights[0][j], single_iter_weights[i][j]
                ), f"{single_iter_weights[0][j]} vs. {single_iter_weights[i][j]}"
        weights_per_device.append(single_iter_weights[0])
    return weights_per_device


def compare_weights(
    W1: List[List[torch.Tensor]],
    W2: List[List[torch.Tensor]],
    desc: str,
    allow_error: bool = False,
) -> None:
    """
    Compare the weights after each iteration across different training approaches.

    Args:
        W1: Weights after each iteration from one approach.
        W2: Weights after each iteration from the other approach.
        desc: Description of approaches
        allow_error: Whether small errors are allowed.
            Small errors are common if one of the approaches uses DDP and the
            other does not.
    """
    assert len(W1) == len(W2)
    # w1, w2 are weights after a single iteration for the 1st and 2nd approaches,
    # respectively.
    max_diff = 0
    for w1, w2 in zip(W1, W2):
        assert len(w1) == len(w2)
        # t1, t2 are weights of a single layer.
        for t1, t2 in zip(w1, w2):
            t1 = t1.to("cpu")
            t2 = t2.to("cpu")
            if not allow_error:
                assert torch.allclose(
                    t1, t2
                ), f"{desc} max diff: {torch.max(torch.abs(t1 - t2).flatten())}"
            elif not torch.allclose(t1, t2):
                max_diff = max(max_diff, torch.max(torch.abs(t1 - t2).flatten()))

    if max_diff != 0:
        logger.info(f"{desc} max diff: {max_diff}")


def print_elapses(elapses: List[float], name: str, rank: Optional[int] = None) -> int:
    """
    Print individual elapses and their average.

    Args:
        elapses: List of elapses for all iterations
        name: Name of the approach (Ray DDP, torch, or torch DDP).
        rank: Rank in torch DDP.

    Returns:
        avg: Average elapse without first iteration.
    """

    logger.info(name)
    for i, elapse in enumerate(elapses):
        if rank:
            logger.info(
                f"Iteration: {i}, rank: {rank}, elapse: {secs_to_micros(elapse)} us"
            )
        else:
            logger.info(f"Iteration: {i}, elapse: {secs_to_micros(elapse)} us")
    total = sum(elapses)
    avg = total / len(elapses)
    logger.info(f"Average elapse: {secs_to_micros(avg)} us")
    total -= elapses[0]
    avg = total / (len(elapses) - 1)
    avg = secs_to_micros(avg)
    logger.info(f"Average elapse without iteration 0: {avg} us")
    return avg


def main(config: Config) -> None:
    """
    Run and compare the performance of Ray DDP, PyTorch, and PyTorch DDP.
    Correctness of Ray DDP is checked if specified. Save the average elapses
    across iterations of all approaches to the output file.

    Args:
        config: Model and training configurations, as well as whether to check
            correctness and the output file path.
    """
    ray_ddp_weights, ray_ddp_elapse = run_ray_ddp(config)
    torch_weights, torch_elapse = run_torch(config)
    torch_ddp_weights, torch_ddp_elapse = run_torch_ddp(config)
    if config.check_correctness:
        compare_weights(
            ray_ddp_weights, torch_weights, "ray ddp vs torch", allow_error=True
        )
        compare_weights(ray_ddp_weights, torch_ddp_weights, "ray ddp vs torch ddp")
    with open(config.output_file, "w") as file:
        file.write("ray-ddp,torch,torch-ddp\n")
        file.write(f"{ray_ddp_elapse},{torch_elapse},{torch_ddp_elapse}\n")


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
        "--num-layers",
        type=int,
        required=True,
        help="number of layers",
    )
    parser.add_argument(
        "--layer-size",
        type=int,
        required=True,
        help="size of a layer (each layer is a square)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=list(str_to_dtype.keys()),
        required=True,
        help="data type of tensors",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        required=True,
        help="number of iterations",
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
        "--check-correctness",
        action="store_true",
        help="whether to check correctness",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="output file path",
    )
    parser.add_argument(
        "--breakdown-performance",
        action="store_true",
        help="whether to print performance breakdown",
    )
    args = parser.parse_args()
    config = Config(
        num_layers=args.num_layers,
        layer_size=args.layer_size,
        dtype=str_to_dtype[args.dtype],
        num_iters=args.num_iters,
        learning_rate=args.learning_rate,
        num_actors=args.num_actors,
        check_correctness=args.check_correctness,
        output_file=args.output_file,
        breakdown_performance=args.breakdown_performance,
    )
    return config


if __name__ == "__main__":
    main(parse_config())


# 1. baseline (identify performance overhead)
# 2. profile
# 3. y axis: throughput/iter; x axis: increasing model size (keeping layers constant)
