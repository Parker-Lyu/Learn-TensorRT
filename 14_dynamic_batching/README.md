# 14 - Dynamic Batching

Goal: run TensorRT inference with dynamic batch sizes and explicit batched buffer layout.

Topics:

- Static batch versus dynamic batch
- Optimization profiles
- `minShapes`, `optShapes`, and `maxShapes`
- Runtime input shape setting
- Batched NCHW input buffers
- Output offset calculation
- Latency and throughput trade-off

Acceptance criteria:

- A TensorRT engine supports batch size 1 through 4.
- C++ code runs different batch sizes with the same engine.
- Input and output offsets are calculated explicitly.
- A benchmark compares batch size 1 and batch size 4.
