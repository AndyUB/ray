# README

## Results

Default cupy nccl communicator works for devices (1,2) but errors for others.

```log
python -m actor.p2p.compiled.cupy.example --name p2p_bench
[INFO example.py:64 run_p2p_bench] Running with CUDA devices 1,2...
2025-05-04 22:06:39,375 INFO worker.py:1888 -- Started a local Ray instance.
2025-05-04 22:06:46,032 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group de743a8a-cb0c-44da-b83f-3e3d624c10ee on actors: [Actor(Actor, 4c5ee12631fbb7a91bf6231901000000), Actor(Actor, 6b6a1884aac80e8f9eba4c5b01000000)]
2025-05-04 22:06:50,553 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=4159753) Actor 1 completed in 5510 ms
2025-05-04 22:06:55,191 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-04 22:06:55,192 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 4c5ee12631fbb7a91bf6231901000000)
2025-05-04 22:06:55,192 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 6b6a1884aac80e8f9eba4c5b01000000)
(Actor pid=4159753) Destructing NCCL group on actor: Actor(Actor, 6b6a1884aac80e8f9eba4c5b01000000)
2025-05-04 22:06:55,637 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-04 22:06:55,637 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 4c5ee12631fbb7a91bf6231901000000)
2025-05-04 22:06:55,638 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 6b6a1884aac80e8f9eba4c5b01000000)
2025-05-04 22:06:55,638 INFO compiled_dag_node.py:2203 -- Teardown complete
(Actor pid=4159750) Actor 0 completed in 4780 ms
(Actor pid=4159750) Destructing NCCL group on actor: Actor(Actor, 4c5ee12631fbb7a91bf6231901000000)
[INFO example.py:64 run_p2p_bench] Running with CUDA devices 2,3...
2025-05-04 22:07:00,126 INFO worker.py:1888 -- Started a local Ray instance.
2025-05-04 22:07:06,911 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group 4ca70180-8639-4942-b48f-a20d73b19aa1 on actors: [Actor(Actor, c373696d42cb905d9ec47e2001000000), Actor(Actor, e8629230a5372fe48e7380ad01000000)]
2025-05-04 22:07:12,222 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
Traceback (most recent call last):
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/dag/compiled_dag_node.py", line 2531, in _execute_until
    result = self._dag_output_fetcher.read(timeout)
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/common.py", line 309, in read
    outputs = self._read_list(timeout)
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/common.py", line 400, in _read_list
    raise e
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/common.py", line 382, in _read_list
    result = c.read(min(remaining_timeout, iteration_timeout))
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/shared_memory_channel.py", line 776, in read
    return self._channel_dict[self._resolve_actor_id()].read(timeout)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/shared_memory_channel.py", line 612, in read
    output = self._buffers[self._next_read_index].read(timeout)
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/shared_memory_channel.py", line 480, in read
    ret = self._worker.get_objects(
          ~~~~~~~~~~~~~~~~~~~~~~~~^
        [self._local_reader_ref], timeout=timeout, return_exceptions=True
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )[0][0]
    ^
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/_private/worker.py", line 904, in get_objects
    ] = self.core_worker.get_objects(
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^
        object_refs,
        ^^^^^^^^^^^^
        timeout_ms,
        ^^^^^^^^^^^
    )
    ^
  File "python/ray/_raylet.pyx", line 3197, in ray._raylet.CoreWorker.get_objects
  File "python/ray/includes/common.pxi", line 106, in ray._raylet.check_status
ray.exceptions.RayChannelTimeoutError: System error: Timed out waiting for object available to read. ObjectID: 00db14a0419b947be8629230a5372fe48e7380ad0100000002e1f505

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/wxdeng/sea-to-bel-torch/actor/p2p/compiled/cupy/example.py", line 79, in <module>
    fire.Fire(main)
    ~~~~~~~~~^^^^^^
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/fire/core.py", line 135, in Fire
    component_trace = _Fire(component, args, parsed_flag_args, context, name)
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/fire/core.py", line 468, in _Fire
    component, remaining_args = _CallAndUpdateTrace(
                                ~~~~~~~~~~~~~~~~~~~^
        component,
        ^^^^^^^^^^
    ...<2 lines>...
        treatment='class' if is_class else 'routine',
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        target=component.__name__)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/fire/core.py", line 684, in _CallAndUpdateTrace
    component = fn(*varargs, **kwargs)
  File "/home/wxdeng/sea-to-bel-torch/actor/p2p/compiled/cupy/example.py", line 73, in main
    run_p2p_bench()
    ~~~~~~~~~~~~~^^
  File "/home/wxdeng/sea-to-bel-torch/actor/p2p/compiled/cupy/example.py", line 66, in run_p2p_bench
    run_p2p()
    ~~~~~~~^^
  File "/home/wxdeng/sea-to-bel-torch/actor/p2p/compiled/cupy/example.py", line 56, in run_p2p
    ray.get(compiled_dag.execute(None))
    ~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/_private/worker.py", line 2794, in get
    return object_refs.get(timeout=timeout)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/compiled_dag_ref.py", line 115, in get
    self._dag._execute_until(
    ~~~~~~~~~~~~~~~~~~~~~~~~^
        self._execution_index, self._channel_index, timeout
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    )
    ^
  File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/dag/compiled_dag_node.py", line 2541, in _execute_until
    raise RayChannelTimeoutError(
    ...<4 lines>...
    ) from e
ray.exceptions.RayChannelTimeoutError: System error: If the execution is expected to take a long time, increase RAY_CGRAPH_get_timeout which is currently 10 seconds. Otherwise, this may indicate that the execution is hanging.
2025-05-04 22:07:22,334 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-04 22:07:22,335 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, c373696d42cb905d9ec47e2001000000)
2025-05-04 22:07:22,335 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, e8629230a5372fe48e7380ad01000000)
(Actor pid=4180653) Destructing NCCL group on actor: Actor(Actor, e8629230a5372fe48e7380ad01000000)
(Actor pid=4180653) ERROR:root:Compiled DAG task exited with exception
(Actor pid=4180653) Traceback (most recent call last):
(Actor pid=4180653)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/dag/compiled_dag_node.py", line 253, in do_exec_tasks
(Actor pid=4180653)     done = tasks[operation.exec_task_idx].exec_operation(
(Actor pid=4180653)         self, operation.type, overlap_gpu_communication
(Actor pid=4180653)     )
(Actor pid=4180653)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/dag/compiled_dag_node.py", line 780, in exec_operation
(Actor pid=4180653)     return self._read(overlap_gpu_communication)
(Actor pid=4180653)            ~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Actor pid=4180653)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/dag/compiled_dag_node.py", line 672, in _read
(Actor pid=4180653)     input_data = self.input_reader.read()
(Actor pid=4180653)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/common.py", line 309, in read
(Actor pid=4180653)     outputs = self._read_list(timeout)
(Actor pid=4180653)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/common.py", line 382, in _read_list
(Actor pid=4180653)     result = c.read(min(remaining_timeout, iteration_timeout))
(Actor pid=4180653)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/torch_tensor_nccl_channel.py", line 328, in read
(Actor pid=4180653)     tensors = self._gpu_data_channel.read(timeout)
(Actor pid=4180653)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/torch_tensor_nccl_channel.py", line 618, in read
(Actor pid=4180653)     buf = self._nccl_group.recv(
(Actor pid=4180653)         meta.shape, meta.dtype, self._writer_rank, _torch_zeros_allocator
(Actor pid=4180653)     )
(Actor pid=4180653)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/nccl_group.py", line 234, in recv
(Actor pid=4180653)     self._comm.recv(
(Actor pid=4180653)     ~~~~~~~~~~~~~~~^
(Actor pid=4180653)         self.nccl_util.get_tensor_ptr(buf),
(Actor pid=4180653)         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Actor pid=4180653)     ...<3 lines>...
(Actor pid=4180653)         self._recv_stream.ptr,
(Actor pid=4180653)         ^^^^^^^^^^^^^^^^^^^^^^
(Actor pid=4180653)     )
(Actor pid=4180653)     ^
(Actor pid=4180653)   File "cupy_backends/cuda/libs/nccl.pyx", line 481, in cupy_backends.cuda.libs.nccl.NcclCommunicator.recv
(Actor pid=4180653)   File "cupy_backends/cuda/libs/nccl.pyx", line 129, in cupy_backends.cuda.libs.nccl.check_status
(Actor pid=4180653) cupy_backends.cuda.libs.nccl.NcclError: NCCL_ERROR_INTERNAL_ERROR: internal error - please report this issue to the NCCL developers
(Actor pid=4180678)     return self._write()
(Actor pid=4180678)            ~~~~~~~~~~~^^
(Actor pid=4180678)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/dag/compiled_dag_node.py", line 751, in _write
(Actor pid=4180678)     self.output_writer.write(output_val)
(Actor pid=4180678)     ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^
(Actor pid=4180678)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/common.py", line 617, in write
(Actor pid=4180678)     channel.write(val_i, timeout)
(Actor pid=4180678)     ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
(Actor pid=4180678)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/torch_tensor_nccl_channel.py", line 270, in write
(Actor pid=4180678)     self._send_cpu_and_gpu_data(value, timeout)
(Actor pid=4180678)     ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^
(Actor pid=4180678)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/torch_tensor_nccl_channel.py", line 203, in _send_cpu_and_gpu_data
(Actor pid=4180678)     self._gpu_data_channel.write(gpu_tensors)
(Actor pid=4180678)     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^
(Actor pid=4180678)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/torch_tensor_nccl_channel.py", line 573, in write
(Actor pid=4180678)     self._nccl_group.send(tensor, rank)
(Actor pid=4180678)     ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
(Actor pid=4180678)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/experimental/channel/nccl_group.py", line 187, in send
(Actor pid=4180678)     self._comm.send(
(Actor pid=4180678)         self._send_stream.ptr,
(Actor pid=4180678)   File "cupy_backends/cuda/libs/nccl.pyx", line 472, in cupy_backends.cuda.libs.nccl.NcclCommunicator.send
2025-05-04 22:07:23,408 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-04 22:07:23,409 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, c373696d42cb905d9ec47e2001000000)
2025-05-04 22:07:23,409 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, e8629230a5372fe48e7380ad01000000)
2025-05-04 22:07:23,409 INFO compiled_dag_node.py:2203 -- Teardown complete
(Actor pid=4180678) Destructing NCCL group on actor: Actor(Actor, c373696d42cb905d9ec47e2001000000)
(Actor pid=4180678) ERROR:root:Compiled DAG task exited with exception
(Actor pid=4180678) Traceback (most recent call last):
(Actor pid=4180678)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/dag/compiled_dag_node.py", line 253, in do_exec_tasks
(Actor pid=4180678)     done = tasks[operation.exec_task_idx].exec_operation(
(Actor pid=4180678)         self, operation.type, overlap_gpu_communication
(Actor pid=4180678)     ) [repeated 2x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)
(Actor pid=4180678)   File "/home/wxdeng/miniconda3/envs/sea-to-bel-torch/lib/python3.13/site-packages/ray/dag/compiled_dag_node.py", line 786, in exec_operation
(Actor pid=4180678)     ~~~~~~~~~~~~~~~^
(Actor pid=4180678)         self.nccl_util.get_tensor_ptr(buf),
(Actor pid=4180678)         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(Actor pid=4180678)     ...<3 lines>...
(Actor pid=4180678)         ^^^^^^^^^^^^^^^^^^^^^^
(Actor pid=4180678)     ^
(Actor pid=4180678)   File "cupy_backends/cuda/libs/nccl.pyx", line 129, in cupy_backends.cuda.libs.nccl.check_status
(Actor pid=4180678) cupy_backends.cuda.libs.nccl.NcclError: NCCL_ERROR_INTERNAL_ERROR: internal error - please report this issue to the NCCL developers
```
