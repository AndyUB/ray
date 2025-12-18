# README

## Results

The fastest pair (1,2) takes about 3 secs, and (2,3) about 20 secs.
Others take about 40 secs.

```log
python -m actor.p2p.compiled.distributed.example --name p2p_bench
[INFO example.py:165 run_p2p] Running with CUDA devices 0,1...
2025-05-05 18:28:45,129 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=1938145) Initializing process group for actor 0...
(Actor pid=1938148) Initializing process group for actor 1...
(Actor pid=1938145) Process group for actor 0 initialized in 1353 ms
(Actor pid=1938148) Process group for actor 1 initialized in 1305 ms
2025-05-05 18:28:53,105 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 2f7e2f41-2c71-4167-8b7a-5c94a9e3d972 on actors: [Actor(Actor, 39c0c01c2a754b7195c4040f01000000), Actor(Actor, 1ebb240b762bfe2715028ccd01000000)]
(Actor pid=1938145) Initializing communicator for rank 0...
2025-05-05 18:28:53,657 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1938148) Initializing communicator for rank 1...
(Actor pid=1938145) Actor 0 completed in 33740 ms
(Actor pid=1938148) Actor 1 completed in 19206 ms
2025-05-05 18:29:27,507 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:29:27,508 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 39c0c01c2a754b7195c4040f01000000)
2025-05-05 18:29:27,508 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 1ebb240b762bfe2715028ccd01000000)
2025-05-05 18:29:27,527 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:29:27,528 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 39c0c01c2a754b7195c4040f01000000)
2025-05-05 18:29:27,528 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 1ebb240b762bfe2715028ccd01000000)
2025-05-05 18:29:27,528 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:165 run_p2p] Running with CUDA devices 1,2...
2025-05-05 18:29:33,716 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=1961275) Initializing process group for actor 1...
(Actor pid=1961282) Initializing process group for actor 0...
(Actor pid=1961275) Process group for actor 1 initialized in 1565 ms
(Actor pid=1961282) Process group for actor 0 initialized in 1553 ms
2025-05-05 18:29:42,174 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 1822ea02-4448-4e35-bb4f-bc10b1cf2061 on actors: [Actor(Actor, 310c6f7776a7aa4404ed117201000000), Actor(Actor, a718975416593460549dd94001000000)]
(Actor pid=1961282) Initializing communicator for rank 0...
2025-05-05 18:29:42,728 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1961275) Initializing communicator for rank 1...
(Actor pid=1961275) Actor 1 completed in 3703 ms
(Actor pid=1961282) Actor 0 completed in 3950 ms
2025-05-05 18:29:45,331 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:29:45,331 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 310c6f7776a7aa4404ed117201000000)
2025-05-05 18:29:45,331 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, a718975416593460549dd94001000000)
2025-05-05 18:29:45,346 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:29:45,346 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 310c6f7776a7aa4404ed117201000000)
2025-05-05 18:29:45,346 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, a718975416593460549dd94001000000)
2025-05-05 18:29:45,346 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:165 run_p2p] Running with CUDA devices 2,3...
2025-05-05 18:29:50,560 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=1980949) Initializing process group for actor 1...
(Actor pid=1981264) Initializing process group for actor 0...
(Actor pid=1980949) Process group for actor 1 initialized in 1575 ms
(Actor pid=1981264) Process group for actor 0 initialized in 1582 ms
2025-05-05 18:29:59,609 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 6d0e9e36-2043-4e1d-8398-7a44c5057dcf on actors: [Actor(Actor, d43e8482ce50f396d22c485701000000), Actor(Actor, 21e79990a6a9ecf6706bfc0d01000000)]
(Actor pid=1981264) Initializing communicator for rank 0...
2025-05-05 18:30:00,176 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1980949) Initializing communicator for rank 1...
(Actor pid=1980949) Actor 1 completed in 13005 ms
(Actor pid=1981264) Actor 0 completed in 22209 ms
2025-05-05 18:30:21,061 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:30:21,062 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, d43e8482ce50f396d22c485701000000)
2025-05-05 18:30:21,062 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 21e79990a6a9ecf6706bfc0d01000000)
2025-05-05 18:30:21,081 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:30:21,082 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, d43e8482ce50f396d22c485701000000)
2025-05-05 18:30:21,082 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 21e79990a6a9ecf6706bfc0d01000000)
2025-05-05 18:30:21,082 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:165 run_p2p] Running with CUDA devices 3,4...
2025-05-05 18:30:26,494 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2003246) Initializing process group for actor 0...
(Actor pid=2003169) Initializing process group for actor 1...
(Actor pid=2003246) Process group for actor 0 initialized in 1444 ms
(Actor pid=2003169) Process group for actor 1 initialized in 1400 ms
2025-05-05 18:30:35,489 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 297c5e24-5cba-4793-94e1-670bbd563143 on actors: [Actor(Actor, 9f20a7214c44fecf3945cde601000000), Actor(Actor, 8150684bafef55b61ba7158b01000000)]
(Actor pid=2003246) Initializing communicator for rank 0...
2025-05-05 18:30:36,039 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2003169) Initializing communicator for rank 1...
(Actor pid=2003246) Actor 0 completed in 49034 ms
(Actor pid=2003169) Actor 1 completed in 26276 ms
2025-05-05 18:31:23,877 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:31:23,878 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 9f20a7214c44fecf3945cde601000000)
2025-05-05 18:31:23,878 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 8150684bafef55b61ba7158b01000000)
2025-05-05 18:31:23,894 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:31:23,894 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 9f20a7214c44fecf3945cde601000000)
2025-05-05 18:31:23,894 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 8150684bafef55b61ba7158b01000000)
2025-05-05 18:31:23,894 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:165 run_p2p] Running with CUDA devices 4,5...
2025-05-05 18:31:27,950 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2027628) Initializing process group for actor 0...
(Actor pid=2027623) Initializing process group for actor 1...
(Actor pid=2027628) Process group for actor 0 initialized in 1708 ms
(Actor pid=2027623) Process group for actor 1 initialized in 1866 ms
2025-05-05 18:31:37,011 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group dce498c4-fb3b-4103-83c4-8be4c91836fa on actors: [Actor(Actor, 1df517111a5a1ff17ddbd2c201000000), Actor(Actor, 5ab254eeb97a27f544dd9c6001000000)]
(Actor pid=2027628) Initializing communicator for rank 0...
2025-05-05 18:31:37,572 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2027623) Initializing communicator for rank 1...
(Actor pid=2027628) Actor 0 completed in 42873 ms
(Actor pid=2027623) Actor 1 completed in 23582 ms
2025-05-05 18:32:18,832 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:32:18,833 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 1df517111a5a1ff17ddbd2c201000000)
2025-05-05 18:32:18,833 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 5ab254eeb97a27f544dd9c6001000000)
2025-05-05 18:32:18,848 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:32:18,848 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 1df517111a5a1ff17ddbd2c201000000)
2025-05-05 18:32:18,849 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 5ab254eeb97a27f544dd9c6001000000)
2025-05-05 18:32:18,849 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:165 run_p2p] Running with CUDA devices 5,6...
2025-05-05 18:32:24,093 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2050968) Initializing process group for actor 0...
(Actor pid=2050986) Initializing process group for actor 1...
(Actor pid=2050968) Process group for actor 0 initialized in 1560 ms
(Actor pid=2050986) Process group for actor 1 initialized in 1557 ms
2025-05-05 18:32:32,342 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group c5ae06e6-d65f-4f46-a8d1-b920fb595656 on actors: [Actor(Actor, ea4e0ce3752586349be799b901000000), Actor(Actor, 52c3cc6bec2a725a6250d44001000000)]
(Actor pid=2050968) Initializing communicator for rank 0...
2025-05-05 18:32:32,904 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2050986) Initializing communicator for rank 1...
(Actor pid=2050968) Actor 0 completed in 42983 ms
(Actor pid=2050986) Actor 1 completed in 23565 ms
2025-05-05 18:33:14,571 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:33:14,572 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, ea4e0ce3752586349be799b901000000)
2025-05-05 18:33:14,572 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 52c3cc6bec2a725a6250d44001000000)
2025-05-05 18:33:14,590 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:33:14,590 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, ea4e0ce3752586349be799b901000000)
2025-05-05 18:33:14,590 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 52c3cc6bec2a725a6250d44001000000)
2025-05-05 18:33:14,590 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:165 run_p2p] Running with CUDA devices 6,7...
2025-05-05 18:33:21,141 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2076089) Initializing process group for actor 1...
(Actor pid=2076117) Initializing process group for actor 0...
(Actor pid=2076117) Process group for actor 0 initialized in 1415 ms
(Actor pid=2076089) Process group for actor 1 initialized in 1432 ms
2025-05-05 18:33:28,760 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 2a1448be-f01a-492e-bf7c-7b40ccfe8864 on actors: [Actor(Actor, 76558b32d7af76e8cb2749bf01000000), Actor(Actor, 61945c994f22a3f90bbf16cb01000000)]
(Actor pid=2076117) Initializing communicator for rank 0...
2025-05-05 18:33:29,310 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2076089) Initializing communicator for rank 1...
(Actor pid=2076089) Actor 1 completed in 16342 ms
(Actor pid=2076117) Actor 0 completed in 28893 ms
2025-05-05 18:33:57,017 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:33:57,018 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 76558b32d7af76e8cb2749bf01000000)
2025-05-05 18:33:57,018 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 61945c994f22a3f90bbf16cb01000000)
2025-05-05 18:33:57,038 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:33:57,038 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 76558b32d7af76e8cb2749bf01000000)
2025-05-05 18:33:57,038 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 61945c994f22a3f90bbf16cb01000000)
2025-05-05 18:33:57,039 INFO compiled_dag_node.py:2203 -- Teardown complete
```
