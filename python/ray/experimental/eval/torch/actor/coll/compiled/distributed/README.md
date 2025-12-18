# README

## Results

###

Fix ray dag coll api.
Coll group (0,1,2,3) takes about 30 secs, while (4,5,6,7) takes about 75 secs.

```log
python -m actor.coll.compiled.distributed.example --name coll_bench_w4
[INFO example.py:176 run_coll] Running with CUDA devices 0,1,2,3...
2025-05-05 01:09:57,391 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2636567) Initializing process group for actor 1...
(Actor pid=2636562) Initializing process group for actor 0...
(Actor pid=2636597) Initializing process group for actor 3...
(Actor pid=2636611) Initializing process group for actor 2...
(Actor pid=2636597) Process group for actor 3 initialized in 4358 ms
(Actor pid=2636567) Process group for actor 1 initialized in 5104 ms
(Actor pid=2636562) Process group for actor 0 initialized in 5103 ms
(Actor pid=2636611) Process group for actor 2 initialized in 5075 ms
2025-05-05 01:10:09,438 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 71800d66-62d9-4c35-99e7-14601812504d on actors: [Actor(Actor, ef3768f6ab6dbccb877c482501000000), Actor(Actor, 38142d347ead4991e060ea6001000000), Actor(Actor, 3b2f1856f3a42df24a92db8301000000), Actor(Actor, 882327526468bc55c54331ed01000000)]
(Actor pid=2636562) Initializing communicator for rank 0...
2025-05-05 01:10:10,006 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2636567) Initializing communicator for rank 1...
(Actor pid=2636597) Initializing communicator for rank 3...
(Actor pid=2636611) Initializing communicator for rank 2...
(Actor pid=2636562) Actor 0 completed in 32910 ms
(Actor pid=2636567) Actor 1 completed in 33003 ms
(Actor pid=2636597) Actor 3 completed in 32848 ms
(Actor pid=2636611) Actor 2 completed in 32914 ms
2025-05-05 01:10:38,008 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 01:10:38,010 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, ef3768f6ab6dbccb877c482501000000)
2025-05-05 01:10:38,010 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 38142d347ead4991e060ea6001000000)
2025-05-05 01:10:38,010 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 3b2f1856f3a42df24a92db8301000000)
2025-05-05 01:10:38,010 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 882327526468bc55c54331ed01000000)
2025-05-05 01:10:38,035 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 01:10:38,035 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, ef3768f6ab6dbccb877c482501000000)
2025-05-05 01:10:38,036 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 38142d347ead4991e060ea6001000000)
2025-05-05 01:10:38,036 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 3b2f1856f3a42df24a92db8301000000)
2025-05-05 01:10:38,036 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 882327526468bc55c54331ed01000000)
2025-05-05 01:10:38,036 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:176 run_coll] Running with CUDA devices 4,5,6,7...
2025-05-05 01:10:43,548 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2658396) Initializing process group for actor 3...
(Actor pid=2658395) Initializing process group for actor 1...
(Actor pid=2658394) Initializing process group for actor 0...
(Actor pid=2658393) Initializing process group for actor 2...
(Actor pid=2658393) Process group for actor 2 initialized in 3251 ms
(Actor pid=2658396) Process group for actor 3 initialized in 4387 ms
(Actor pid=2658394) Process group for actor 0 initialized in 4876 ms
(Actor pid=2658395) Process group for actor 1 initialized in 5109 ms
2025-05-05 01:10:53,191 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 01f671d7-8ba3-4dca-b148-429d78df1f74 on actors: [Actor(Actor, f039103d324ed0e53bca029c01000000), Actor(Actor, e5d8758cbd4611ab4f2f47fc01000000), Actor(Actor, 835ee0f4986828b0abd3dc0501000000), Actor(Actor, 0383d4a31679f160d32d6afa01000000)]
(Actor pid=2658394) Initializing communicator for rank 0...
2025-05-05 01:10:53,773 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2658395) Initializing communicator for rank 1...
(Actor pid=2658396) Initializing communicator for rank 3...
(Actor pid=2658393) Initializing communicator for rank 2...
(Actor pid=2658394) Actor 0 completed in 76310 ms
(Actor pid=2658395) Actor 1 completed in 76344 ms
(Actor pid=2658396) Actor 3 completed in 76371 ms
(Actor pid=2658393) Actor 2 completed in 76249 ms
2025-05-05 01:12:05,195 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 01:12:05,196 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, f039103d324ed0e53bca029c01000000)
2025-05-05 01:12:05,196 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, e5d8758cbd4611ab4f2f47fc01000000)
2025-05-05 01:12:05,196 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 835ee0f4986828b0abd3dc0501000000)
2025-05-05 01:12:05,196 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 0383d4a31679f160d32d6afa01000000)
2025-05-05 01:12:05,223 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 01:12:05,224 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, f039103d324ed0e53bca029c01000000)
2025-05-05 01:12:05,224 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, e5d8758cbd4611ab4f2f47fc01000000)
2025-05-05 01:12:05,224 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 835ee0f4986828b0abd3dc0501000000)
2025-05-05 01:12:05,224 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 0383d4a31679f160d32d6afa01000000)
2025-05-05 01:12:05,224 INFO compiled_dag_node.py:2203 -- Teardown complete
```

###

Directly use torch dist coll instead of ray dag coll api.
Coll group (0,1,2,3) takes about 30 secs, while (4,5,6,7) takes about 70 secs.

```log
python -m actor.coll.compiled.distributed.example --name coll_bench_w4
[INFO example.py:176 run_coll] Running with CUDA devices 0,1,2,3...
2025-05-05 00:56:29,354 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2462082) Initializing process group for actor 0...
(Actor pid=2462465) Initializing process group for actor 2...
(Actor pid=2462474) Initializing process group for actor 3...
(Actor pid=2462443) Initializing process group for actor 1...
(Actor pid=2462082) Process group for actor 0 initialized in 3788 ms
(Actor pid=2462443) Process group for actor 1 initialized in 3645 ms
(Actor pid=2462474) Process group for actor 3 initialized in 3693 ms
(Actor pid=2462465) Process group for actor 2 initialized in 3693 ms
(Actor pid=2462082) Running coll for actor 0...
(Actor pid=2462443) Running coll for actor 1...
(Actor pid=2462474) Running coll for actor 3...
(Actor pid=2462465) Running coll for actor 2...
(Actor pid=2462082) Worker 0 reduced tensor: tensor([6.], device='cuda:0') in 25478 ms
(Actor pid=2462082) Actor 0 completed in 29629 ms
(Actor pid=2462443) Worker 1 reduced tensor: tensor([6.], device='cuda:0') in 25478 ms
(Actor pid=2462443) Actor 1 completed in 29568 ms
(Actor pid=2462474) Worker 3 reduced tensor: tensor([6.], device='cuda:0') in 25465 ms
(Actor pid=2462474) Actor 3 completed in 29590 ms
(Actor pid=2462465) Worker 2 reduced tensor: tensor([6.], device='cuda:0') in 25479 ms
(Actor pid=2462465) Actor 2 completed in 29613 ms
2025-05-05 00:57:06,692 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 00:57:06,693 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, f8dfaf8090e0d365bf7280d201000000)
2025-05-05 00:57:06,693 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 8831ba047f978db486ce965f01000000)
2025-05-05 00:57:06,693 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, e2df98d0c18b74f85a75075401000000)
2025-05-05 00:57:06,693 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 0b9cd7952bd0fb01d483f25201000000)
2025-05-05 00:57:06,703 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 00:57:06,703 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, f8dfaf8090e0d365bf7280d201000000)
2025-05-05 00:57:06,704 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 8831ba047f978db486ce965f01000000)
2025-05-05 00:57:06,704 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, e2df98d0c18b74f85a75075401000000)
2025-05-05 00:57:06,704 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 0b9cd7952bd0fb01d483f25201000000)
2025-05-05 00:57:06,704 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:176 run_coll] Running with CUDA devices 4,5,6,7...
2025-05-05 00:57:13,798 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2483408) Initializing process group for actor 0...
(Actor pid=2483431) Initializing process group for actor 1...
(Actor pid=2483476) Initializing process group for actor 2...
(Actor pid=2483464) Initializing process group for actor 3...
(Actor pid=2483408) Process group for actor 0 initialized in 4567 ms
(Actor pid=2483431) Process group for actor 1 initialized in 4538 ms
(Actor pid=2483476) Process group for actor 2 initialized in 4562 ms
(Actor pid=2483464) Process group for actor 3 initialized in 4695 ms
(Actor pid=2483408) Running coll for actor 0...
(Actor pid=2483431) Running coll for actor 1...
(Actor pid=2483476) Running coll for actor 2...
(Actor pid=2483464) Running coll for actor 3...
(Actor pid=2483408) Worker 0 reduced tensor: tensor([6.], device='cuda:0') in 63716 ms
(Actor pid=2483476) Worker 2 reduced tensor: tensor([6.], device='cuda:0') in 63720 ms
(Actor pid=2483464) Worker 3 reduced tensor: tensor([6.], device='cuda:0') in 63716 ms
(Actor pid=2483408) Actor 0 completed in 68794 ms
(Actor pid=2483431) Worker 1 reduced tensor: tensor([6.], device='cuda:0') in 63717 ms
(Actor pid=2483431) Actor 1 completed in 68792 ms
(Actor pid=2483476) Actor 2 completed in 68795 ms
(Actor pid=2483464) Actor 3 completed in 68787 ms
2025-05-05 00:58:29,802 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 00:58:29,803 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 64f6fec75383ccc3377325a201000000)
2025-05-05 00:58:29,803 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, f89b3f648b95adf2588a94ce01000000)
2025-05-05 00:58:29,803 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 47d59b28db5b6bd264b7793201000000)
2025-05-05 00:58:29,803 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, e9a647cdc1ae40a45ed8a8de01000000)
2025-05-05 00:58:29,816 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 00:58:29,816 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 64f6fec75383ccc3377325a201000000)
2025-05-05 00:58:29,816 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, f89b3f648b95adf2588a94ce01000000)
2025-05-05 00:58:29,816 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 47d59b28db5b6bd264b7793201000000)
2025-05-05 00:58:29,817 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, e9a647cdc1ae40a45ed8a8de01000000)
2025-05-05 00:58:29,817 INFO compiled_dag_node.py:2203 -- Teardown complete
```
