# 04 - CUDA Memory And Stream

## Purpose

- Learn the CUDA concepts needed for TensorRT inference code.

TensorRT inference code is mostly resource orchestration:

```text
CPU input tensor
  -> host-to-device copy
  -> enqueue inference on a CUDA stream
  -> device-to-host copy for outputs
  -> synchronize only when CPU code needs the result
```

If the memory type or synchronization point is wrong, the model may still produce correct output,
but latency and throughput can be much worse. This lesson uses a tiny CUDA kernel as a stand-in for
TensorRT inference so the memory behavior is easy to inspect before adding TensorRT APIs.

## Prerequisites

- Complete `00_environment_check` in the pinned development container.
- An accessible NVIDIA GPU is required for the runnable CUDA artifact.

## Deliverables

- `cuda_memory_stream` executable with focused CUDA memory-flow helpers
- Explicit-copy, mapped-memory, and managed-memory execution modes
- Per-mode correctness and CUDA-event timing output

## Data Flow

The program creates a synthetic `float32` tensor shaped like one YOLO input:

```text
[1, 3, 640, 640] = 1,228,800 float values
```

It then compares four paths:

- Pageable host memory: `std::vector<float>` plus `cudaMemcpyAsync`.
- Pinned host memory: `cudaMallocHost` plus `cudaMemcpyAsync`.
- Mapped pinned memory: `cudaHostAllocMapped`, no explicit copy, kernel accesses host memory through
  a device pointer.
- Unified Memory: `cudaMallocManaged` with explicit prefetch before the measured kernel loop.

CUDA 13 changed `cudaMemPrefetchAsync` from an integer destination to an explicit
`cudaMemLocation` plus flags. The lesson keeps the compatibility detail in two small helpers so the
data-flow example remains readable while still compiling against the pinned CUDA 13.0 toolkit.

Each path runs:

```text
input[i] -> output[i] = input[i] * 2 + 1
```

The output is validated on the CPU so this lesson checks correctness, not just timing.

## Directory Layout

- `CMakeLists.txt`: target-based CUDA/C++17 build file for this runnable lesson.
- `include/cuda_memory_demo.hpp`: small public config and lesson entry point.
- `src/main.cpp`: command-line parsing and error reporting.
- `include/cuda_resources.hpp`: RAII owners for CUDA memory, streams, and events.
- `src/cuda_memory_demo.cu`: memory allocation paths, kernel launch, timing, and validation.

The code keeps CUDA resources behind small RAII classes so later lessons can extend the same habit
toward TensorRT engines, execution contexts, streams, and buffers.

## Visual Mental Model

If you are new to CUDA, read this section before the bullet list below. The important idea is not
"copy is slow" or "streams are magic"; it is that CPU memory, GPU memory, and queued GPU work live
behind different boundaries.

### One Inference-Like Stream

```mermaid
sequenceDiagram
    autonumber
    participant CPU as CPU code
    participant Stream as CUDA stream queue
    participant Copy as GPU copy engine
    participant SM as GPU kernel engine

    CPU->>Stream: enqueue H2D copy
    CPU->>Stream: enqueue kernel launch
    CPU->>Stream: enqueue D2H copy
    Note over CPU,Stream: CPU returns after queueing work unless it reaches a sync boundary.

    Stream->>Copy: run H2D copy
    Copy-->>Stream: copy finished
    Stream->>SM: run kernel
    SM-->>Stream: kernel finished
    Stream->>Copy: run D2H copy
    Copy-->>Stream: output copy finished

    CPU->>Stream: cudaStreamSynchronize
    Stream-->>CPU: output is now safe to read on CPU
```

The stream is an ordered queue. Work submitted to the same stream runs in order, so TensorRT-style
code usually queues input copy, inference, and output copy on one stream, then synchronizes only
when CPU code really needs the output.

### Memory Choices

```mermaid
flowchart TB
    subgraph Host["CPU host memory"]
        Pageable["Pageable memory<br/>std::vector<float>"]
        Pinned["Pinned memory<br/>cudaMallocHost"]
        Mapped["Mapped pinned memory<br/>cudaHostAllocMapped"]
        Managed["Unified Memory<br/>cudaMallocManaged"]
    end

    subgraph GPU["GPU side"]
        DeviceIn["Device input buffer<br/>cudaMalloc"]
        DeviceOut["Device output buffer<br/>cudaMalloc"]
        Kernel["Kernel or TensorRT enqueue"]
    end

    Pageable -->|"may need hidden staging first"| Staging["Temporary pinned staging buffer"]
    Staging -->|"DMA over PCIe"| DeviceIn
    Pinned -->|"async DMA over PCIe"| DeviceIn
    DeviceIn --> Kernel
    Kernel --> DeviceOut
    DeviceOut -->|"async DMA over PCIe"| Pinned

    Mapped -->|"device pointer aliases host pages"| Kernel
    Kernel -->|"reads/writes host memory over PCIe"| Mapped

    Managed -->|"prefetch migrates pages before timing"| DeviceIn
    DeviceOut -->|"pages may migrate back when CPU reads"| Managed
```

This lesson compares four ways to feed the same simple kernel. The output can be correct in all
paths, but the data movement cost and synchronization behavior can be very different.

## Key Takeaways

- `cudaMemcpyAsync` only behaves like a useful asynchronous transfer when the host memory is pinned.
- Pageable memory can require internal staging before the GPU transfer starts.
- Mapped pinned memory avoids an explicit `cudaMemcpy`, but on a discrete GPU the kernel still reads
  and writes host memory over PCIe.
- Unified Memory makes ownership simple, but page migration can add latency unless access patterns
  are controlled with prefetching or careful warmup.
- CUDA events measure GPU work queued in a stream; CPU timers measure a different boundary.
- In TensorRT code, call `enqueue` and copies on the same stream, then synchronize only where the CPU
  must consume the output.

## Build

```bash
cmake -S 04_cuda_memory_stream -B 04_cuda_memory_stream/build
cmake --build 04_cuda_memory_stream/build
```

## Run

Run with the default tensor size and 20 measured iterations (compares all four memory paths):

```bash
./04_cuda_memory_stream/build/cuda_memory_stream
```

Current GPU output is not available for the 2026-10-27 revision: the Ubuntu SSH connection
timed out and Docker is unavailable on the Mac. CUDA compilation and execution were not performed.
Earlier numeric examples used different measurement code and are omitted to avoid treating them as
current results.

Run a faster smoke test (262,144 elements and five iterations):

```bash
./04_cuda_memory_stream/build/cuda_memory_stream 262144 5
```

Use a larger buffer to make transfer cost easier to see (4,194,304 elements):

```bash
./04_cuda_memory_stream/build/cuda_memory_stream 4194304 20
```


## Outputs

The program prints:

- CUDA device name
- whether mapped host memory is supported
- buffer size
- the timed scope for each path
- average CUDA-event time and host wall time per iteration
- effective logical read/write rate for explicit-copy and mapped paths
- CPU validation result

Exact numbers depend on GPU, PCIe generation, CPU memory speed, current clocks, and whether the
first run paid extra initialization cost.

## Timing boundaries

| Path | Event interval | Host interval |
| --- | --- | --- |
| Pageable / pinned | H2D + kernel + D2H | Before recording start through waiting for stop |
| Mapped pinned | Kernel accessing mapped host memory | Before recording start through waiting for stop |
| Managed | Kernel after prefetch and warmup | Before recording start through waiting for stop |

All paths complete one untimed warmup. The host clock is `std::chrono::steady_clock`; each
column divides the complete repeated interval by the iteration count, so it is a batch average,
not a per-request latency distribution. Host time includes submission and waiting overhead.
Subtracting event time from host time does not isolate pure CPU overhead.

Managed prefetch in both directions, allocation, initialization, CPU validation, and cleanup are
outside both timers. A small managed time is not evidence that Unified Memory is the fastest
end-to-end strategy. A fair end-to-end comparison needs the same initial data location and final
CPU-result boundary for every path; this lesson intentionally shows distinct memory-access scopes.

The effective rate is two logical buffer sizes divided by event time. It is not isolated PCIe
bandwidth or a measurement of actual device-memory traffic. Single-stream H2D, kernel, and D2H
remain ordered; this demo does not demonstrate copy/compute overlap.

## Resource and validation notes

Moves transfer ownership with `std::exchange`; the memory wrapper requires a nonthrowing deleter
and keeps cleanup nonthrowing. Stream destruction waits before dependent resources unwind, with
cleanup errors reported to stderr. The internal transform treats count zero as a no-op; the demo
CLI still requires a positive count. Output validation rejects non-finite values.

The 2026-10-22 cleanup update was reviewed statically on macOS. CUDA compilation, GPU execution,
and timing were not rerun; earlier example output is not validation of the modified implementation.

### Measurement and validation update (2026-10-26)

Each path now completes one untimed warmup before its event interval. Managed prefetch remains
outside the timed kernel interval. Reported effective rate divides logical read/write bytes by
the whole measured interval; it is not isolated copy bandwidth. Earlier example timings predate
this change and require a GPU rerun before any before/after comparison.
`include/transform_checks.hpp` holds the host helpers also used by the demo: safe ceil division
for block count, checked float byte sizing, and rejection of non-finite expected/output values.
The empty internal launch remains a no-op; the CLI rejects zero elements. Host boundary checks do
not execute GPU threads or prove device bounds correctness. CUDA source changes are static-review
only in this local pass.

### Timing and readability update (2026-10-27)

CUDA resource owners moved unchanged into `include/cuda_resources.hpp`. The explicit-copy helper
now fixes its H2D direction internally. Output names each timed scope and reports event and host
averages. Managed result migration remains outside both timers. These changes received static
review only: the Ubuntu SSH connection timed out, and local Docker is unavailable. No new CUDA
compilation, GPU correctness, sanitizer, timing, or speedup result is claimed.

## Tests

Host-only block-count, byte-overflow, and transform-validation tests need no GPU. To build only these checks in the usual development environment:

```bash
cmake -S 04_cuda_memory_stream -B 04_cuda_memory_stream/build-cpu -DCPU_ONLY_TESTS=ON
cmake --build 04_cuda_memory_stream/build-cpu --target transform_checks_tests
ctest --test-dir 04_cuda_memory_stream/build-cpu --output-on-failure
```

2026-10-26 local-only verification: Apple Clang 21.0.0 compiled the CPU test directly as ISO C++17
with `-Wall -Wextra -Wpedantic -Werror`; it passed. A separate AddressSanitizer/UndefinedBehaviorSanitizer
build also passed without reports. Docker/CMake and CUDA/TensorRT were unavailable on the Mac;
these CMake commands and GPU execution were not run during this review.

## Checkpoints

- Explain why the event/host difference is not pure CPU overhead, and why managed time cannot
  be ranked against the full explicit-copy flow as an end-to-end result.
- Change `iterations` from `5` to `50` and observe whether average time stabilizes.
- Compare default input size with `4194304` elements and explain which path becomes bandwidth-bound.
- Remove pinned memory from the explicit-copy path mentally: explain why overlap with inference would
  become unreliable.
- Explain why mapped pinned memory can be attractive for tiny metadata but risky for large image
  tensors on a discrete GPU.


## Appendix: CUDA Function Qualifiers

CUDA uses function qualifiers to say where a function runs and who is allowed to call it. In this
lesson, `transform_kernel` is marked with `__global__` because CPU code launches it onto the GPU
stream.

| Qualifier | Runs on | Called from | Return type | Typical use |
| --- | --- | --- | --- | --- |
| `__global__` | GPU device | Host code, and on supported devices also device code with dynamic parallelism | Must return `void` | Kernel entry points launched with `<<<grid, block, shared_memory, stream>>>`. This is the boundary where CPU code queues GPU work. |
| `__device__` | GPU device | Device code | Any supported device-callable type | Helper functions used by kernels, such as math, indexing, coordinate transforms, or small reusable operations. |
| `__host__` | CPU host | Host code | Normal C++ return types | Ordinary CPU functions. CUDA treats unqualified C++ functions as host functions by default. |
| `__host__ __device__` | CPU host and GPU device | Host code and device code | Any type valid in both compilation modes | Small utilities that should compile for both sides, as long as they avoid APIs or language features unavailable on the device. |

The important distinction is that `__global__` describes a kernel launch boundary, while
`__device__` describes GPU-only helper code. A `__global__` function is queued from the CPU and runs
many threads on the GPU; a `__device__` function is called by those GPU threads like a normal helper.

## Appendix: CUDA Allocation Choices

| API | Allocates | Key characteristics | Typical use in inference code |
| --- | --- | --- | --- |
| `cudaMalloc` | Device memory | Memory lives in GPU device memory. CPU code cannot directly dereference the pointer. Data must be moved with copies such as `cudaMemcpyAsync`, or produced by GPU work. | Tensor input/output bindings, intermediate device buffers, and any buffer that kernels or TensorRT enqueue calls access repeatedly. |
| `cudaMallocHost` | Pinned host memory | Allocates page-locked CPU memory. It costs more to allocate than normal pageable memory, but enables efficient DMA transfers and useful `cudaMemcpyAsync` behavior. Free with `cudaFreeHost`. | Reusable CPU staging buffers for input upload and output download, especially when copies should overlap with GPU work on streams. |
| `cudaHostAlloc` | Pinned host memory with selectable flags | More configurable pinned allocation. It can allocate ordinary pinned memory, mapped pinned memory, portable pinned memory, or write-combined pinned memory depending on flags. Free with `cudaFreeHost`. | Special host buffers when you need mapped host access, cross-context portability, write-mostly upload staging, or a combination of these behaviors. |
| `cudaMallocManaged` | Unified Memory | Allocates one managed pointer visible to CPU and GPU. Pages migrate between processors on demand or through explicit `cudaMemPrefetchAsync`. Convenience is high, but uncontrolled migration can add latency. Free with `cudaFree`. | Teaching, prototypes, irregular data structures, or workloads where simpler ownership matters more than tightly controlled latency. Use prefetching before latency-sensitive GPU work. |

`cudaHostAlloc` flag summary:

| Flag | Meaning | Notes |
| --- | --- | --- |
| `cudaHostAllocDefault` | Default pinned host allocation | Similar intent to `cudaMallocHost`: page-locked host memory suitable for faster transfers. |
| `cudaHostAllocMapped` | Maps the host allocation into the device address space | Use `cudaHostGetDevicePointer` to get the device-visible pointer. On a discrete GPU, kernel reads and writes still travel over PCIe, so this is usually better for small or sparse access than large image tensors. |
| `cudaHostAllocPortable` | Makes the pinned allocation portable across CUDA contexts | Useful in multi-context programs. Most simple single-device lessons do not need it. |
| `cudaHostAllocWriteCombined` | Allocates write-combined host memory | Can improve CPU-to-GPU upload bandwidth for CPU write-only staging, but CPU reads from this memory are slow. Avoid it for buffers the CPU must read back often. |

Flags can be combined with bitwise OR when the combination makes sense, for example
`cudaHostAllocMapped | cudaHostAllocPortable`.
