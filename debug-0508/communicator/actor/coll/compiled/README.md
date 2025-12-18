# README

## Results

###

Both cupy and torch comm work fine.

```log
python -m actor.coll.compiled.example --name coll_bench_w2 --comm create
[INFO example.py:176 run_coll] Running with CUDA devices 0,1...
2025-05-05 18:45:49,390 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2224763) Initializing process group for actor 1...
(Actor pid=2224771) Initializing process group for actor 0...
(Actor pid=2224771) Process group for actor 0 initialized in 2166 ms
(Actor pid=2224763) Process group for actor 1 initialized in 2323 ms
2025-05-05 18:45:57,562 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group 6f7647b3-4321-4bad-a05a-654a990e2011 on actors: [Actor(Actor, 8237d6cad52cde89cfd803ff01000000), Actor(Actor, f7c3ca2cc27c04268fd3081401000000)]
2025-05-05 18:45:58,708 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2224763) Actor 1 completed in 26545 ms
(Actor pid=2224771) Actor 0 completed in 26559 ms
2025-05-05 18:46:22,570 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:46:22,571 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 8237d6cad52cde89cfd803ff01000000)
2025-05-05 18:46:22,571 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, f7c3ca2cc27c04268fd3081401000000)
(Actor pid=2224763) Destructing NCCL group on actor: Actor(Actor, f7c3ca2cc27c04268fd3081401000000)
(Actor pid=2224771) Destructing NCCL group on actor: Actor(Actor, 8237d6cad52cde89cfd803ff01000000)
2025-05-05 18:46:23,192 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:46:23,192 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 8237d6cad52cde89cfd803ff01000000)
2025-05-05 18:46:23,193 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, f7c3ca2cc27c04268fd3081401000000)
2025-05-05 18:46:23,193 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:176 run_coll] Running with CUDA devices 1,2...
2025-05-05 18:46:28,957 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2247122) Initializing process group for actor 0...
(Actor pid=2247116) Initializing process group for actor 1...
(Actor pid=2247116) Process group for actor 1 initialized in 1558 ms
(Actor pid=2247122) Process group for actor 0 initialized in 1661 ms
2025-05-05 18:46:37,537 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group 27f656d3-49ea-4f47-adad-8a797c5f5367 on actors: [Actor(Actor, 52c593228b6dca0d12d984de01000000), Actor(Actor, 66631072129f65a1158d20c601000000)]
2025-05-05 18:46:38,638 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2247116) Actor 1 completed in 3476 ms
(Actor pid=2247122) Actor 0 completed in 3582 ms
2025-05-05 18:46:40,268 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:46:40,269 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 52c593228b6dca0d12d984de01000000)
2025-05-05 18:46:40,269 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 66631072129f65a1158d20c601000000)
(Actor pid=2247116) Destructing NCCL group on actor: Actor(Actor, 66631072129f65a1158d20c601000000)
(Actor pid=2247122) Destructing NCCL group on actor: Actor(Actor, 52c593228b6dca0d12d984de01000000)
2025-05-05 18:46:41,042 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:46:41,043 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 52c593228b6dca0d12d984de01000000)
2025-05-05 18:46:41,043 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 66631072129f65a1158d20c601000000)
2025-05-05 18:46:41,043 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:176 run_coll] Running with CUDA devices 2,3...
2025-05-05 18:46:45,356 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2267391) Initializing process group for actor 1...
(Actor pid=2267388) Initializing process group for actor 0...
(Actor pid=2267391) Process group for actor 1 initialized in 1348 ms
(Actor pid=2267388) Process group for actor 0 initialized in 1366 ms
2025-05-05 18:46:53,202 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group e1fa661a-0a5c-4456-b1e9-31e2ab9fa7af on actors: [Actor(Actor, 8b51cfe6adf5532a1f075ab301000000), Actor(Actor, e2ba9ad7d06d170f31a4f97901000000)]
2025-05-05 18:46:54,215 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2267391) Actor 1 completed in 16176 ms
(Actor pid=2267388) Actor 0 completed in 16253 ms
2025-05-05 18:47:08,818 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:47:08,819 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 8b51cfe6adf5532a1f075ab301000000)
2025-05-05 18:47:08,819 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, e2ba9ad7d06d170f31a4f97901000000)
(Actor pid=2267391) Destructing NCCL group on actor: Actor(Actor, e2ba9ad7d06d170f31a4f97901000000)
(Actor pid=2267388) Destructing NCCL group on actor: Actor(Actor, 8b51cfe6adf5532a1f075ab301000000)
2025-05-05 18:47:09,163 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:47:09,164 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 8b51cfe6adf5532a1f075ab301000000)
2025-05-05 18:47:09,164 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, e2ba9ad7d06d170f31a4f97901000000)
2025-05-05 18:47:09,164 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:176 run_coll] Running with CUDA devices 3,4...
2025-05-05 18:47:15,180 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2289684) Initializing process group for actor 0...
(Actor pid=2289680) Initializing process group for actor 1...
(Actor pid=2289684) Process group for actor 0 initialized in 1484 ms
(Actor pid=2289680) Process group for actor 1 initialized in 1401 ms
2025-05-05 18:47:23,106 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group c51a0078-84cb-42f3-99af-e8503f5b531f on actors: [Actor(Actor, b7da17e5e51747281d178e5f01000000), Actor(Actor, beafce21abc64a89572415ca01000000)]
2025-05-05 18:47:24,226 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2289684) Actor 0 completed in 37362 ms
(Actor pid=2289680) Actor 1 completed in 37279 ms
2025-05-05 18:47:59,792 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:47:59,793 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, b7da17e5e51747281d178e5f01000000)
2025-05-05 18:47:59,793 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, beafce21abc64a89572415ca01000000)
(Actor pid=2289684) Destructing NCCL group on actor: Actor(Actor, b7da17e5e51747281d178e5f01000000)
(Actor pid=2289680) Destructing NCCL group on actor: Actor(Actor, beafce21abc64a89572415ca01000000)
2025-05-05 18:48:00,266 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:48:00,266 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, b7da17e5e51747281d178e5f01000000)
2025-05-05 18:48:00,267 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, beafce21abc64a89572415ca01000000)
2025-05-05 18:48:00,267 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:176 run_coll] Running with CUDA devices 4,5...
2025-05-05 18:48:05,051 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2314228) Initializing process group for actor 0...
(Actor pid=2314247) Initializing process group for actor 1...
(Actor pid=2314247) Process group for actor 1 initialized in 1206 ms
(Actor pid=2314228) Process group for actor 0 initialized in 1530 ms
2025-05-05 18:48:12,002 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group d03e9593-8757-405e-b7b2-806be095764b on actors: [Actor(Actor, 027214b01ed942914d167a9101000000), Actor(Actor, bfdb01ca8a5b3a7a0af0589801000000)]
2025-05-05 18:48:12,886 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2314228) Actor 0 completed in 35738 ms
(Actor pid=2314247) Actor 1 completed in 35730 ms
2025-05-05 18:48:47,020 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:48:47,022 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 027214b01ed942914d167a9101000000)
2025-05-05 18:48:47,022 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, bfdb01ca8a5b3a7a0af0589801000000)
(Actor pid=2314228) Destructing NCCL group on actor: Actor(Actor, 027214b01ed942914d167a9101000000)
(Actor pid=2314247) Destructing NCCL group on actor: Actor(Actor, bfdb01ca8a5b3a7a0af0589801000000)
2025-05-05 18:48:47,483 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:48:47,484 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 027214b01ed942914d167a9101000000)
2025-05-05 18:48:47,484 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, bfdb01ca8a5b3a7a0af0589801000000)
2025-05-05 18:48:47,484 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:176 run_coll] Running with CUDA devices 5,6...
2025-05-05 18:48:53,423 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2339388) Initializing process group for actor 1...
(Actor pid=2339392) Initializing process group for actor 0...
(Actor pid=2339392) Process group for actor 0 initialized in 1828 ms
(Actor pid=2339388) Process group for actor 1 initialized in 1899 ms
2025-05-05 18:49:01,971 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group 8a1c5927-83dc-4404-9bef-94b0fb96f074 on actors: [Actor(Actor, 988e45339598ea8e3476b00401000000), Actor(Actor, e96579a38203ddd760748c5a01000000)]
2025-05-05 18:49:02,991 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2339392) Actor 0 completed in 32586 ms
(Actor pid=2339388) Actor 1 completed in 32647 ms
2025-05-05 18:49:33,524 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:49:33,524 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 988e45339598ea8e3476b00401000000)
2025-05-05 18:49:33,524 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, e96579a38203ddd760748c5a01000000)
(Actor pid=2339392) Destructing NCCL group on actor: Actor(Actor, 988e45339598ea8e3476b00401000000)
(Actor pid=2339388) Destructing NCCL group on actor: Actor(Actor, e96579a38203ddd760748c5a01000000)
2025-05-05 18:49:34,154 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:49:34,155 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 988e45339598ea8e3476b00401000000)
2025-05-05 18:49:34,155 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, e96579a38203ddd760748c5a01000000)
2025-05-05 18:49:34,155 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:176 run_coll] Running with CUDA devices 6,7...
2025-05-05 18:49:40,110 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2364518) Initializing process group for actor 0...
(Actor pid=2364511) Initializing process group for actor 1...
(Actor pid=2364518) Process group for actor 0 initialized in 1668 ms
(Actor pid=2364511) Process group for actor 1 initialized in 1708 ms
2025-05-05 18:49:47,102 INFO torch_tensor_nccl_channel.py:772 -- Creating NCCL group da4713d0-3104-40a7-8c17-b4bc292986a1 on actors: [Actor(Actor, 962dd09d4e8b25f07ca4625a01000000), Actor(Actor, 57a34cb6dbde506bf7a2569001000000)]
2025-05-05 18:49:48,394 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2364518) Actor 0 completed in 22570 ms
(Actor pid=2364511) Actor 1 completed in 22578 ms
2025-05-05 18:50:08,767 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:50:08,768 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 962dd09d4e8b25f07ca4625a01000000)
2025-05-05 18:50:08,768 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 57a34cb6dbde506bf7a2569001000000)
(Actor pid=2364518) Destructing NCCL group on actor: Actor(Actor, 962dd09d4e8b25f07ca4625a01000000)
(Actor pid=2364511) Destructing NCCL group on actor: Actor(Actor, 57a34cb6dbde506bf7a2569001000000)
2025-05-05 18:50:09,687 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:50:09,688 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 962dd09d4e8b25f07ca4625a01000000)
2025-05-05 18:50:09,688 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 57a34cb6dbde506bf7a2569001000000)
2025-05-05 18:50:09,688 INFO compiled_dag_node.py:2203 -- Teardown complete

python -m actor.coll.compiled.example --name coll_bench_w2 --comm distributed
[INFO example.py:176 run_coll] Running with CUDA devices 0,1...
2025-05-05 18:51:57,376 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2400966) Initializing process group for actor 0...
(Actor pid=2400956) Initializing process group for actor 1...
(Actor pid=2400956) Process group for actor 1 initialized in 1179 ms
(Actor pid=2400966) Process group for actor 0 initialized in 1348 ms
2025-05-05 18:52:06,167 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group b7d9fa66-a5a1-4b2c-b4ca-1919fc8372c6 on actors: [Actor(Actor, 3b95c64b8c778c2a83e59d1601000000), Actor(Actor, 50a30a96f2ea4e057cca3b7501000000)]
(Actor pid=2400966) Initializing communicator for rank 0...
2025-05-05 18:52:06,738 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2400956) Initializing communicator for rank 1...
(Actor pid=2400966) Actor 0 completed in 26567 ms
(Actor pid=2400956) Actor 1 completed in 26443 ms
2025-05-05 18:52:32,178 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:52:32,179 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 3b95c64b8c778c2a83e59d1601000000)
2025-05-05 18:52:32,179 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 50a30a96f2ea4e057cca3b7501000000)
2025-05-05 18:52:32,202 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:52:32,202 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 3b95c64b8c778c2a83e59d1601000000)
2025-05-05 18:52:32,203 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 50a30a96f2ea4e057cca3b7501000000)
2025-05-05 18:52:32,203 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:176 run_coll] Running with CUDA devices 1,2...
2025-05-05 18:52:38,262 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2427070) Initializing process group for actor 1...
(Actor pid=2427067) Initializing process group for actor 0...
(Actor pid=2427070) Process group for actor 1 initialized in 2008 ms
(Actor pid=2427067) Process group for actor 0 initialized in 1990 ms
2025-05-05 18:52:47,798 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 4f9e4820-dbe8-4659-8497-772799d08a7f on actors: [Actor(Actor, d91c14a9b52e63ec7d34da0401000000), Actor(Actor, fb46672c64ce089d6986d28501000000)]
2025-05-05 18:52:48,347 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2427070) Initializing communicator for rank 1...
(Actor pid=2427067) Initializing communicator for rank 0...
(Actor pid=2427070) Actor 1 completed in 4406 ms
(Actor pid=2427067) Actor 0 completed in 4316 ms
2025-05-05 18:52:50,904 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:52:50,905 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, d91c14a9b52e63ec7d34da0401000000)
2025-05-05 18:52:50,905 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, fb46672c64ce089d6986d28501000000)
2025-05-05 18:52:50,919 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:52:50,920 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, d91c14a9b52e63ec7d34da0401000000)
2025-05-05 18:52:50,920 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, fb46672c64ce089d6986d28501000000)
2025-05-05 18:52:50,920 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:176 run_coll] Running with CUDA devices 2,3...
2025-05-05 18:52:56,866 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2447619) Initializing process group for actor 1...
(Actor pid=2447635) Initializing process group for actor 0...
(Actor pid=2447619) Process group for actor 1 initialized in 1795 ms
(Actor pid=2447635) Process group for actor 0 initialized in 1802 ms
2025-05-05 18:53:05,248 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group fdaa98d4-799a-4fcb-b2ef-b094afe84423 on actors: [Actor(Actor, 351357b744249de90f9e961c01000000), Actor(Actor, 884594ebea552acbbe9de78a01000000)]
(Actor pid=2447635) Initializing communicator for rank 0...
2025-05-05 18:53:05,859 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2447619) Initializing communicator for rank 1...
(Actor pid=2447619) Actor 1 completed in 17642 ms
(Actor pid=2447635) Actor 0 completed in 17627 ms
2025-05-05 18:53:21,861 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:53:21,862 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 351357b744249de90f9e961c01000000)
2025-05-05 18:53:21,862 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 884594ebea552acbbe9de78a01000000)
2025-05-05 18:53:21,879 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:53:21,880 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 351357b744249de90f9e961c01000000)
2025-05-05 18:53:21,880 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 884594ebea552acbbe9de78a01000000)
2025-05-05 18:53:21,880 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:176 run_coll] Running with CUDA devices 3,4...
2025-05-05 18:53:26,086 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2469862) Initializing process group for actor 0...
(Actor pid=2469880) Initializing process group for actor 1...
(Actor pid=2469862) Process group for actor 0 initialized in 1497 ms
(Actor pid=2469880) Process group for actor 1 initialized in 1519 ms
2025-05-05 18:53:32,641 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 7a1931aa-0077-46c5-8f83-48096e207bb2 on actors: [Actor(Actor, bcd38e7250c7350d0e71b61201000000), Actor(Actor, d5bbbfff0a77e68355a97e0a01000000)]
(Actor pid=2469862) Initializing communicator for rank 0...
2025-05-05 18:53:33,205 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2469880) Initializing communicator for rank 1...
(Actor pid=2469862) Actor 0 completed in 38160 ms
(Actor pid=2469880) Actor 1 completed in 38151 ms
2025-05-05 18:54:10,059 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:54:10,060 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, bcd38e7250c7350d0e71b61201000000)
2025-05-05 18:54:10,060 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, d5bbbfff0a77e68355a97e0a01000000)
2025-05-05 18:54:10,078 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:54:10,078 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, bcd38e7250c7350d0e71b61201000000)
2025-05-05 18:54:10,079 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, d5bbbfff0a77e68355a97e0a01000000)
2025-05-05 18:54:10,079 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:176 run_coll] Running with CUDA devices 4,5...
2025-05-05 18:54:16,135 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2494004) Initializing process group for actor 1...
(Actor pid=2494029) Initializing process group for actor 0...
(Actor pid=2494004) Process group for actor 1 initialized in 1719 ms
(Actor pid=2494029) Process group for actor 0 initialized in 1642 ms
2025-05-05 18:54:24,568 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group eed05a0e-021d-4116-93af-562f63e9d42c on actors: [Actor(Actor, d6deafd44052b333fad1262801000000), Actor(Actor, 0623d05832e718df1c8bd99901000000)]
(Actor pid=2494029) Initializing communicator for rank 0...
2025-05-05 18:54:25,138 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2494004) Initializing communicator for rank 1...
(Actor pid=2494004) Actor 1 completed in 34466 ms
(Actor pid=2494029) Actor 0 completed in 34447 ms
2025-05-05 18:54:58,072 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:54:58,073 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, d6deafd44052b333fad1262801000000)
2025-05-05 18:54:58,074 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 0623d05832e718df1c8bd99901000000)
2025-05-05 18:54:58,093 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:54:58,093 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, d6deafd44052b333fad1262801000000)
2025-05-05 18:54:58,094 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 0623d05832e718df1c8bd99901000000)
2025-05-05 18:54:58,094 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:176 run_coll] Running with CUDA devices 5,6...
2025-05-05 18:55:04,700 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2520463) Initializing process group for actor 0...
(Actor pid=2520480) Initializing process group for actor 1...
(Actor pid=2520463) Process group for actor 0 initialized in 1798 ms
(Actor pid=2520480) Process group for actor 1 initialized in 1653 ms
2025-05-05 18:55:12,605 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 1345855f-4e7d-4f82-9b88-0bf76dbeffd4 on actors: [Actor(Actor, fcb2268a3b2ee0a058d55f6001000000), Actor(Actor, b17750f76acd27ee4c7f578401000000)]
(Actor pid=2520463) Initializing communicator for rank 0...
2025-05-05 18:55:13,170 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2520480) Initializing communicator for rank 1...
(Actor pid=2520463) Actor 0 completed in 33380 ms
(Actor pid=2520480) Actor 1 completed in 33245 ms
2025-05-05 18:55:44,983 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:55:44,984 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, fcb2268a3b2ee0a058d55f6001000000)
2025-05-05 18:55:44,984 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, b17750f76acd27ee4c7f578401000000)
2025-05-05 18:55:45,002 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:55:45,003 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, fcb2268a3b2ee0a058d55f6001000000)
2025-05-05 18:55:45,003 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, b17750f76acd27ee4c7f578401000000)
2025-05-05 18:55:45,003 INFO compiled_dag_node.py:2203 -- Teardown complete
[INFO example.py:176 run_coll] Running with CUDA devices 6,7...
2025-05-05 18:55:50,096 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=2544994) Initializing process group for actor 1...
(Actor pid=2544990) Initializing process group for actor 0...
(Actor pid=2544990) Process group for actor 0 initialized in 1687 ms
(Actor pid=2544994) Process group for actor 1 initialized in 1810 ms
2025-05-05 18:55:58,260 INFO torch_tensor_nccl_channel.py:770 -- Initializing custom NCCL group 51b9f0bd-696d-490b-9c2d-dfd516bce739 on actors: [Actor(Actor, 7b4b4d84f30a9678b81c6a5801000000), Actor(Actor, 21c88690cf4c417a76d5707101000000)]
(Actor pid=2544990) Initializing communicator for rank 0...
2025-05-05 18:55:58,813 INFO torch_tensor_nccl_channel.py:797 -- NCCL group initialized.
(Actor pid=2544994) Initializing communicator for rank 1...
(Actor pid=2544994) Actor 1 completed in 22540 ms
(Actor pid=2544990) Actor 0 completed in 22457 ms
2025-05-05 18:56:19,778 INFO compiled_dag_node.py:2173 -- Tearing down compiled DAG
2025-05-05 18:56:19,779 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 7b4b4d84f30a9678b81c6a5801000000)
2025-05-05 18:56:19,779 INFO compiled_dag_node.py:2178 -- Cancelling compiled worker on actor: Actor(Actor, 21c88690cf4c417a76d5707101000000)
2025-05-05 18:56:19,798 INFO compiled_dag_node.py:2200 -- Waiting for worker tasks to exit
2025-05-05 18:56:19,799 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 7b4b4d84f30a9678b81c6a5801000000)
2025-05-05 18:56:19,799 INFO compiled_dag_node.py:2161 -- Killing actor: Actor(Actor, 21c88690cf4c417a76d5707101000000)
2025-05-05 18:56:19,799 INFO compiled_dag_node.py:2203 -- Teardown complete
```

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
