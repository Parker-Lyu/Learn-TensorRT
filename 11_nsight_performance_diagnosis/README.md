# 11 - Nsight Performance Diagnosis

Goal: use timeline evidence to diagnose and explain inference bottlenecks.

Topics:

- `trtexec` baseline
- `nsys` command-line capture
- Nsight Systems timeline reading
- CPU preprocessing bottleneck
- H2D and D2H copy gaps
- GPU starvation
- CUDA stream overlap verification
- P50/P90/P99 latency reporting

Acceptance criteria:

- A C++ TensorRT program is profiled with Nsight Systems.
- The report identifies whether the GPU is busy or waiting.
- One optimization is explained with before-and-after evidence.
