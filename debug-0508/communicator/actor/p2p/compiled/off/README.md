# README

## Results

When w/o compile, all pairs take about 5 secs.

```log
python -m actor.p2p.compiled.example --name p2p_bench
[INFO example.py:57 run_p2p_bench] Running with CUDA devices 0,1...
2025-05-04 21:39:57,832 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=3752256) Actor 1 completed in 6786 ms
(Actor pid=3752260) Actor 0 completed in 2360 ms
[INFO example.py:57 run_p2p_bench] Running with CUDA devices 1,2...
2025-05-04 21:40:17,325 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=3771305) Actor 1 completed in 4351 ms
(Actor pid=3771309) Actor 0 completed in 2246 ms
[INFO example.py:57 run_p2p_bench] Running with CUDA devices 2,3...
2025-05-04 21:40:35,339 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=3791360) Actor 0 completed in 1961 ms
(Actor pid=3791396) Actor 1 completed in 3920 ms
[INFO example.py:57 run_p2p_bench] Running with CUDA devices 3,4...
2025-05-04 21:40:53,717 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=3811160) Actor 0 completed in 1779 ms
(Actor pid=3811173) Actor 1 completed in 3846 ms
[INFO example.py:57 run_p2p_bench] Running with CUDA devices 4,5...
2025-05-04 21:41:11,261 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=3829747) Actor 0 completed in 1534 ms
(Actor pid=3829791) Actor 1 completed in 3491 ms
[INFO example.py:57 run_p2p_bench] Running with CUDA devices 5,6...
2025-05-04 21:41:27,075 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=3849188) Actor 1 completed in 3674 ms
(Actor pid=3849219) Actor 0 completed in 1542 ms
[INFO example.py:57 run_p2p_bench] Running with CUDA devices 6,7...
2025-05-04 21:41:43,634 INFO worker.py:1888 -- Started a local Ray instance.
(Actor pid=3868968) Actor 0 completed in 2310 ms
(Actor pid=3868974) Actor 1 completed in 4307 ms
```
