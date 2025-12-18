# README

## Results

Pair (1,2) takes about 3 secs, (2,3) takes about 20 secs, others take about 40 secs.

```log
python -m actor.p2p.compiled.cupy.example --name p2p_bench
[INFO example.py:47 run_p2p] Running with CUDA devices 0,1...
2025-05-05 18:11:38,947 INFO worker.py:1888 -- Started a local Ray instance.
2025-05-05 18:11:45,341 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group ecfa870e-ea06-42db-ac4b-799f6f58306b on actors: [Actor(Actor, ef2e0d670bc689143b3c1cf401000000), Actor(Actor, 752f17bd665ea86fddefc0b901000000)]
2025-05-05 18:11:47,642 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1699695) Actor 0 completed in 31825 ms
(Actor pid=1699715) Actor 1 completed in 17539 ms
2025-05-05 18:12:18,872 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:12:18,873 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, ef2e0d670bc689143b3c1cf401000000)
2025-05-05 18:12:18,873 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 752f17bd665ea86fddefc0b901000000)
(Actor pid=1699695) Destructing NCCL group on actor: Actor(Actor, ef2e0d670bc689143b3c1cf401000000)
(Actor pid=1699715) Destructing NCCL group on actor: Actor(Actor, 752f17bd665ea86fddefc0b901000000)
2025-05-05 18:12:19,502 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:12:19,502 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, ef2e0d670bc689143b3c1cf401000000)
2025-05-05 18:12:19,502 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 752f17bd665ea86fddefc0b901000000)
2025-05-05 18:12:19,503 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:47 run_p2p] Running with CUDA devices 1,2...
2025-05-05 18:12:24,054 INFO worker.py:1888 -- Started a local Ray instance.
2025-05-05 18:12:30,427 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group e38494b2-7856-46bc-bc71-c12b2e79ef5d on actors: [Actor(Actor, d385417f31cb31699ee726bc01000000), Actor(Actor, 8a2f32f2572c1c436800af9801000000)]
2025-05-05 18:12:32,857 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1721470) Actor 0 completed in 3585 ms
(Actor pid=1721469) Actor 1 completed in 3399 ms
2025-05-05 18:12:34,844 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:12:34,845 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, d385417f31cb31699ee726bc01000000)
2025-05-05 18:12:34,845 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 8a2f32f2572c1c436800af9801000000)
(Actor pid=1721470) Destructing NCCL group on actor: Actor(Actor, d385417f31cb31699ee726bc01000000)
(Actor pid=1721469) Destructing NCCL group on actor: Actor(Actor, 8a2f32f2572c1c436800af9801000000)
2025-05-05 18:12:35,743 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:12:35,744 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, d385417f31cb31699ee726bc01000000)
2025-05-05 18:12:35,744 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 8a2f32f2572c1c436800af9801000000)
2025-05-05 18:12:35,744 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:47 run_p2p] Running with CUDA devices 2,3...
2025-05-05 18:12:40,486 INFO worker.py:1888 -- Started a local Ray instance.
2025-05-05 18:12:46,453 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group f14fcc6e-f4af-4ce1-a9fa-74d745edf71f on actors: [Actor(Actor, 7178d631e240e306bfc1701001000000), Actor(Actor, 1e5364dc40fb523bfff95c6f01000000)]
2025-05-05 18:12:48,968 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1740887) Actor 1 completed in 12025 ms
(Actor pid=1740893) Actor 0 completed in 21101 ms
2025-05-05 18:13:08,396 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:13:08,397 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 1e5364dc40fb523bfff95c6f01000000)
2025-05-05 18:13:08,397 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 7178d631e240e306bfc1701001000000)
(Actor pid=1740887) Destructing NCCL group on actor: Actor(Actor, 7178d631e240e306bfc1701001000000)
(Actor pid=1740893) Destructing NCCL group on actor: Actor(Actor, 1e5364dc40fb523bfff95c6f01000000)
2025-05-05 18:13:09,096 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:13:09,096 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 1e5364dc40fb523bfff95c6f01000000)
2025-05-05 18:13:09,097 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 7178d631e240e306bfc1701001000000)
2025-05-05 18:13:09,097 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:47 run_p2p] Running with CUDA devices 3,4...
2025-05-05 18:13:14,464 INFO worker.py:1888 -- Started a local Ray instance.
2025-05-05 18:13:21,077 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group 5a1ac3cb-9319-4e0d-8e78-773637c8a1c7 on actors: [Actor(Actor, a8990d1c22ff8c04eb163b3e01000000), Actor(Actor, e0af27440c780825b3eec7d001000000)]
2025-05-05 18:13:23,397 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1762682) Actor 0 completed in 48612 ms
(Actor pid=1762705) Actor 1 completed in 25795 ms
2025-05-05 18:14:10,519 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:14:10,520 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, e0af27440c780825b3eec7d001000000)
2025-05-05 18:14:10,520 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, a8990d1c22ff8c04eb163b3e01000000)
(Actor pid=1762682) Destructing NCCL group on actor: Actor(Actor, e0af27440c780825b3eec7d001000000)
(Actor pid=1762705) Destructing NCCL group on actor: Actor(Actor, a8990d1c22ff8c04eb163b3e01000000)
2025-05-05 18:14:10,997 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:14:10,997 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, e0af27440c780825b3eec7d001000000)
2025-05-05 18:14:10,997 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, a8990d1c22ff8c04eb163b3e01000000)
2025-05-05 18:14:10,997 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:47 run_p2p] Running with CUDA devices 4,5...
2025-05-05 18:14:15,308 INFO worker.py:1888 -- Started a local Ray instance.
2025-05-05 18:14:21,981 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group 8cd47cbd-2f41-4dba-90da-a45ddd586f07 on actors: [Actor(Actor, 289cfea9a8c3064754399a6301000000), Actor(Actor, 1b69be6c89e3766c7c0ed20201000000)]
2025-05-05 18:14:24,371 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1786807) Actor 0 completed in 42035 ms
(Actor pid=1786831) Actor 1 completed in 22942 ms
2025-05-05 18:15:04,808 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:15:04,808 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 1b69be6c89e3766c7c0ed20201000000)
2025-05-05 18:15:04,808 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 289cfea9a8c3064754399a6301000000)
(Actor pid=1786807) Destructing NCCL group on actor: Actor(Actor, 1b69be6c89e3766c7c0ed20201000000)
(Actor pid=1786831) Destructing NCCL group on actor: Actor(Actor, 289cfea9a8c3064754399a6301000000)
2025-05-05 18:15:05,447 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:15:05,447 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 1b69be6c89e3766c7c0ed20201000000)
2025-05-05 18:15:05,447 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 289cfea9a8c3064754399a6301000000)
2025-05-05 18:15:05,448 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:47 run_p2p] Running with CUDA devices 5,6...
2025-05-05 18:15:09,689 INFO worker.py:1888 -- Started a local Ray instance.
2025-05-05 18:15:15,924 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group a2e95160-ef51-4b83-9174-ad1df9b9c6a4 on actors: [Actor(Actor, d9ea488af2d8f05652d1a31101000000), Actor(Actor, fb03522c8750869f38857d0f01000000)]
2025-05-05 18:15:18,449 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1810460) Actor 1 completed in 22220 ms
(Actor pid=1810471) Actor 0 completed in 41964 ms
2025-05-05 18:15:58,709 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:15:58,710 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, fb03522c8750869f38857d0f01000000)
2025-05-05 18:15:58,710 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, d9ea488af2d8f05652d1a31101000000)
(Actor pid=1810460) Destructing NCCL group on actor: Actor(Actor, d9ea488af2d8f05652d1a31101000000)
(Actor pid=1810471) Destructing NCCL group on actor: Actor(Actor, fb03522c8750869f38857d0f01000000)
2025-05-05 18:15:59,399 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:15:59,399 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, fb03522c8750869f38857d0f01000000)
2025-05-05 18:15:59,399 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, d9ea488af2d8f05652d1a31101000000)
2025-05-05 18:15:59,399 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:47 run_p2p] Running with CUDA devices 6,7...
2025-05-05 18:16:04,294 INFO worker.py:1888 -- Started a local Ray instance.
2025-05-05 18:16:10,619 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group 64cbc1d2-6e3e-4254-98e5-181126f5c861 on actors: [Actor(Actor, 4ce11d8783ecf9901c28e99201000000), Actor(Actor, c8e754b0e5b7868e50428ea901000000)]
2025-05-05 18:16:13,484 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1834771) Actor 0 completed in 30372 ms
(Actor pid=1834779) Actor 1 completed in 17136 ms
2025-05-05 18:16:41,802 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:16:41,802 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 4ce11d8783ecf9901c28e99201000000)
2025-05-05 18:16:41,803 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, c8e754b0e5b7868e50428ea901000000)
(Actor pid=1834771) Destructing NCCL group on actor: Actor(Actor, 4ce11d8783ecf9901c28e99201000000)
(Actor pid=1834779) Destructing NCCL group on actor: Actor(Actor, c8e754b0e5b7868e50428ea901000000)
2025-05-05 18:16:42,513 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:16:42,513 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 4ce11d8783ecf9901c28e99201000000)
2025-05-05 18:16:42,513 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, c8e754b0e5b7868e50428ea901000000)
2025-05-05 18:16:42,513 INFO compiled_dag_node.py:2203 -- Teardown complete
```
