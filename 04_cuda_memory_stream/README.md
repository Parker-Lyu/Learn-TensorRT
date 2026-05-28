# 04 - CUDA Memory And Stream

Goal: learn the CUDA runtime concepts needed by TensorRT C++ inference code.

Topics:

- Device memory allocation
- Pinned host memory with `cudaMallocHost`
- Mapped pinned memory with `cudaHostAllocMapped`
- Unified Memory with `cudaMallocManaged`
- Explicit copy versus mapped access versus managed memory
- Host-to-device and device-to-host copies
- CUDA streams
- CUDA events for timing
- Synchronization cost

Acceptance criteria:

- A small program allocates buffers, copies data asynchronously, and reports timing.
- You can explain when `cudaMemcpyAsync` is actually asynchronous.
- You can explain why mapped pinned memory avoids an explicit copy but still consumes PCIe bandwidth on a discrete GPU.
- You can explain when Unified Memory helps development and when page migration can hurt latency.
