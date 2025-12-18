# README

## Results

This approximates results for `actor.p2p.interpreted`.
The fastest pairs (1,2) take about 5 secs and (2,3) about 30 secs.
Others take about 80 secs.

```log
python -m actor.p2p.compiled.distributed.example --name p2p_bench
[INFO example.py:162 run_p2p] Running with CUDA devices 0,1...
2025-05-04 23:52:24,485 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=1401350) Initializing process group for actor 0...
(Actor pid=1401395) Process group for actor 1 initialized in 2677 ms
2025-05-04 23:52:34,010 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 262f00ff-0bb0-491f-a4aa-c535f5a33010 on actors: [Actor(Actor, eb4835e056257f20787cdc1b01000000), Actor(Actor, 0389fee83ba1eb0938e8cc8901000000)]
(Actor pid=1401350) Initializing communicator for rank 0...
2025-05-04 23:52:34,587 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1401350) Actor 0 completed in 46045 ms
(Actor pid=1401395) Initializing process group for actor 1...
(Actor pid=1401350) Process group for actor 0 initialized in 3007 ms
(Actor pid=1401395) Initializing communicator for rank 1...
2025-05-04 23:53:20,446 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-04 23:53:20,447 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, eb4835e056257f20787cdc1b01000000)
2025-05-04 23:53:20,447 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 0389fee83ba1eb0938e8cc8901000000)
2025-05-04 23:53:20,468 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-04 23:53:20,468 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, eb4835e056257f20787cdc1b01000000)
2025-05-04 23:53:20,469 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 0389fee83ba1eb0938e8cc8901000000)
2025-05-04 23:53:20,469 INFO compiled_dag_node.py:2203 -- Teardown complete
(Actor pid=1401395) Actor 1 completed in 26061 ms
[INFO example.py:162 run_p2p] Running with CUDA devices 1,2...
2025-05-04 23:53:26,333 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=1425776) Initializing process group for actor 0...
(Actor pid=1425776) Process group for actor 0 initialized in 2408 ms
2025-05-04 23:53:35,129 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 0687b553-8675-494a-a689-08739e72acbb on actors: [Actor(Actor, 74abee4c5dd956d96b78470801000000), Actor(Actor, 325cbd78ec7eff71c198c38101000000)]
(Actor pid=1425776) Initializing communicator for rank 0...
2025-05-04 23:53:35,675 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1425776) Actor 0 completed in 5890 ms
(Actor pid=1425800) Initializing process group for actor 1...
2025-05-04 23:53:39,353 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-04 23:53:39,354 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 74abee4c5dd956d96b78470801000000)
2025-05-04 23:53:39,354 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 325cbd78ec7eff71c198c38101000000)
2025-05-04 23:53:39,375 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-04 23:53:39,375 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 74abee4c5dd956d96b78470801000000)
2025-05-04 23:53:39,375 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 325cbd78ec7eff71c198c38101000000)
2025-05-04 23:53:39,376 INFO compiled_dag_node.py:2203 -- Teardown complete
(Actor pid=1425800) Process group for actor 1 initialized in 2515 ms
(Actor pid=1425800) Initializing communicator for rank 1...
(Actor pid=1425800) Actor 1 completed in 5347 ms
[INFO example.py:162 run_p2p] Running with CUDA devices 2,3...
2025-05-04 23:53:44,410 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=1444833) Initializing process group for actor 1...
(Actor pid=1444875) Process group for actor 0 initialized in 3302 ms
2025-05-04 23:53:54,254 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 3e11d311-3b62-4d69-a270-48d4e8380ab9 on actors: [Actor(Actor, 5117cb8a6706454c2f6d161001000000), Actor(Actor, 885c507182ad34c8d35701af01000000)]
(Actor pid=1444875) Initializing communicator for rank 0...
2025-05-04 23:53:54,818 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1444833) Actor 1 completed in 20130 ms
(Actor pid=1444875) Initializing process group for actor 0...
(Actor pid=1444833) Process group for actor 1 initialized in 3429 ms
(Actor pid=1444833) Initializing communicator for rank 1...
2025-05-04 23:54:23,501 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-04 23:54:23,502 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 5117cb8a6706454c2f6d161001000000)
2025-05-04 23:54:23,502 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 885c507182ad34c8d35701af01000000)
2025-05-04 23:54:23,520 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-04 23:54:23,520 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 5117cb8a6706454c2f6d161001000000)
2025-05-04 23:54:23,520 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 885c507182ad34c8d35701af01000000)
2025-05-04 23:54:23,520 INFO compiled_dag_node.py:2203 -- Teardown complete
(Actor pid=1444875) Actor 0 completed in 31794 ms
[INFO example.py:162 run_p2p] Running with CUDA devices 3,4...
2025-05-04 23:54:28,800 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=1467747) Initializing process group for actor 0...
(Actor pid=1467747) Process group for actor 0 initialized in 2862 ms
2025-05-04 23:54:36,873 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 1d237ff8-401b-4b25-acb7-950bc7443050 on actors: [Actor(Actor, 3a76053f37323b06b5a1d50f01000000), Actor(Actor, 7aff8e4affd76dffd3e0721501000000)]
(Actor pid=1467747) Initializing communicator for rank 0...
2025-05-04 23:54:37,440 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1467747) Actor 0 completed in 73498 ms
(Actor pid=1467779) Initializing process group for actor 1...
(Actor pid=1467779) Process group for actor 1 initialized in 2786 ms
(Actor pid=1467779) Initializing communicator for rank 1...
2025-05-04 23:55:48,313 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-04 23:55:48,315 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 3a76053f37323b06b5a1d50f01000000)
2025-05-04 23:55:48,315 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 7aff8e4affd76dffd3e0721501000000)
2025-05-04 23:55:48,332 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-04 23:55:48,333 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 3a76053f37323b06b5a1d50f01000000)
2025-05-04 23:55:48,333 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 7aff8e4affd76dffd3e0721501000000)
2025-05-04 23:55:48,333 INFO compiled_dag_node.py:2203 -- Teardown complete
(Actor pid=1467779) Actor 1 completed in 39728 ms
[INFO example.py:162 run_p2p] Running with CUDA devices 4,5...
2025-05-04 23:55:54,375 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=1497061) Initializing process group for actor 0...
(Actor pid=1497061) Process group for actor 0 initialized in 2283 ms
2025-05-04 23:56:03,884 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 669cff5f-62b0-4158-8c22-58498b45ba3c on actors: [Actor(Actor, 8895931ed6e258d5c4c83a9601000000), Actor(Actor, 10d8da67c004f78c2df0147001000000)]
(Actor pid=1497061) Initializing communicator for rank 0...
2025-05-04 23:56:04,432 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1497061) Actor 0 completed in 79298 ms
(Actor pid=1497144) Initializing process group for actor 1...
(Actor pid=1497144) Process group for actor 1 initialized in 2205 ms
(Actor pid=1497144) Initializing communicator for rank 1...
2025-05-04 23:57:21,743 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-04 23:57:21,744 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 8895931ed6e258d5c4c83a9601000000)
2025-05-04 23:57:21,744 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 10d8da67c004f78c2df0147001000000)
2025-05-04 23:57:21,764 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-04 23:57:21,764 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 8895931ed6e258d5c4c83a9601000000)
2025-05-04 23:57:21,764 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 10d8da67c004f78c2df0147001000000)
2025-05-04 23:57:21,764 INFO compiled_dag_node.py:2203 -- Teardown complete
(Actor pid=1497144) Actor 1 completed in 43283 ms
[INFO example.py:162 run_p2p] Running with CUDA devices 5,6...
2025-05-04 23:57:26,786 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=1528267) Initializing process group for actor 0...
(Actor pid=1528267) Process group for actor 0 initialized in 2934 ms
2025-05-04 23:57:36,070 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 74fb0aec-c758-4f11-b868-cbe330d39541 on actors: [Actor(Actor, e84968cb0e406e5e33e7096101000000), Actor(Actor, e4deb839eeab60eeffe8399701000000)]
(Actor pid=1528267) Initializing communicator for rank 0...
2025-05-04 23:57:36,604 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1528254) Actor 1 completed in 42959 ms
(Actor pid=1528254) Initializing process group for actor 1...
(Actor pid=1528254) Process group for actor 1 initialized in 2936 ms
(Actor pid=1528254) Initializing communicator for rank 1...
2025-05-04 23:58:49,302 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-04 23:58:49,303 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, e84968cb0e406e5e33e7096101000000)
2025-05-04 23:58:49,303 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, e4deb839eeab60eeffe8399701000000)
2025-05-04 23:58:49,323 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-04 23:58:49,323 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, e84968cb0e406e5e33e7096101000000)
2025-05-04 23:58:49,324 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, e4deb839eeab60eeffe8399701000000)
2025-05-04 23:58:49,324 INFO compiled_dag_node.py:2203 -- Teardown complete
(Actor pid=1528267) Actor 0 completed in 75414 ms
[INFO example.py:162 run_p2p] Running with CUDA devices 6,7...
2025-05-04 23:58:55,231 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=1557984) Initializing process group for actor 0...
(Actor pid=1557984) Process group for actor 0 initialized in 2457 ms
2025-05-04 23:59:04,953 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 3509511d-4fe1-4931-92bb-243ed16a28a6 on actors: [Actor(Actor, 4ffd8fa52678f8ff8c43068701000000), Actor(Actor, 4273e9659be366905dda85d801000000)]
(Actor pid=1557984) Initializing communicator for rank 0...
2025-05-04 23:59:05,514 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=1557984) Actor 0 completed in 50159 ms
(Actor pid=1558008) Initializing process group for actor 1...
(Actor pid=1558008) Process group for actor 1 initialized in 2335 ms
(Actor pid=1558008) Initializing communicator for rank 1...
2025-05-04 23:59:53,486 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-04 23:59:53,487 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 4ffd8fa52678f8ff8c43068701000000)
2025-05-04 23:59:53,487 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 4273e9659be366905dda85d801000000)
2025-05-04 23:59:53,502 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-04 23:59:53,502 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 4ffd8fa52678f8ff8c43068701000000)
2025-05-04 23:59:53,503 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 4273e9659be366905dda85d801000000)
2025-05-04 23:59:53,503 INFO compiled_dag_node.py:2203 -- Teardown complete
(Actor pid=1558008) Actor 1 completed in 28988 ms
```
