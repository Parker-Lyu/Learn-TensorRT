# 04 - CUDA Memory And Stream

Goal: learn the CUDA runtime concepts needed by TensorRT C++ inference code.

Topics:

- Device memory allocation
- Pinned host memory with `cudaMallocHost`
- Host-to-device and device-to-host copies
- CUDA streams
- CUDA events for timing
- Synchronization cost

Acceptance criteria:

- A small program allocates buffers, copies data asynchronously, and reports timing.
- You can explain when `cudaMemcpyAsync` is actually asynchronous.
