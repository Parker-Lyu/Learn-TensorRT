# 07 - TensorRT RAII Resource

This lesson turns the TensorRT engine artifact from lesson 06 into a small C++ runtime loader with
explicit resource ownership.

Goal: make TensorRT and CUDA C++ code safe enough to grow toward long-running inference services.

Topics:

- RAII for CUDA streams, CUDA events, device buffers, and pinned host buffers
- `std::unique_ptr` with a custom TensorRT deleter
- TensorRT runtime, engine, and execution context ownership
- Name-based TensorRT IO APIs: `getNbIOTensors`, `getIOTensorName`, `setTensorAddress`, `enqueueV3`
- Exception-safe initialization when file loading, deserialization, allocation, or enqueue fails
- Move-only resource classes
- Dynamic input shape setup for engines that need runtime dimensions
- Deterministic failure injection after engine read, runtime/engine/context creation, first buffer,
  stream creation, or immediately before enqueue
- Repeated create/destroy checks with host RSS and CUDA device-memory snapshots

## Why This Matters

TensorRT inference code is resource-management code before it is model code:

```text
read serialized engine
  -> create TensorRT runtime
  -> deserialize CUDA engine
  -> create execution context
  -> allocate one buffer per IO tensor
  -> bind addresses by tensor name
  -> enqueue inference on a CUDA stream
```

If a service leaks one context, stream, or device buffer per camera reconnect, it can run perfectly
in a demo and still fail after hours of production traffic. RAII makes the cleanup path automatic:
once a C++ object owns a resource, its destructor releases that resource even when initialization
fails halfway through.

## Directory Layout

- `CMakeLists.txt`: target-based C++17 build file that links TensorRT and CUDA Runtime.
- `include/tensorrt_raii.hpp`: small public API for configuring and running the smoke inference.
- `src/tensorrt_raii.cpp`: TensorRT logger, smart-pointer deleter, CUDA RAII wrappers, tensor buffer
  allocation, address binding, and enqueue timing.
- `src/main.cpp`: command-line parsing and concise reporting.
- `tests/config_tests.cpp`: focused validation of failure-stage configuration.

The reusable logic is built as `tensorrt_raii_lib`, with a thin executable named
`tensorrt_raii_resource` on top.

## Prerequisites

Use the TensorRT development container from lesson 00. Generate at least one engine with lesson 06:

```bash
python3 06_trtexec_engine/build_and_benchmark.py --builds static_fp32
```

The default executable path expects:

```text
06_trtexec_engine/outputs/yolov8n_static_fp32.engine
```

Serialized TensorRT engines are machine-local artifacts. Rebuild them inside the target container
and on the target GPU/software stack.

## Build

Run from this lesson directory:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Run

Run the default static FP32 engine from lesson 06:

```bash
./build/tensorrt_raii_resource
```

Use a specific engine:

```bash
./build/tensorrt_raii_resource \
  --engine ../06_trtexec_engine/outputs/yolov8n_static_fp16.engine
```

Run a dynamic engine by supplying runtime input dimensions:

```bash
./build/tensorrt_raii_resource \
  --engine ../06_trtexec_engine/outputs/yolov8n_dynamic_fp16.engine \
  --input-shape images:1x3x640x640
```

Adjust warmup and measurement count:

```bash
./build/tensorrt_raii_resource --warmup 3 --iterations 10
```

Exercise repeated construction/destruction after three unmeasured priming cycles:

```bash
./build/tensorrt_raii_resource --repeat 100
```

Inject a failure after the first tensor buffer has been acquired, repeat it, and verify that already
owned resources are released during stack unwinding:

```bash
./build/tensorrt_raii_resource \
  --repeat 100 \
  --inject-failure first-buffer \
  --memory-tolerance-mib 16
```

Valid stages are `engine-read`, `runtime`, `engine`, `context`, `first-buffer`, `stream`, and
`enqueue`. A failed memory-stability gate returns a nonzero status. The tolerance accounts for host
allocator and driver caching; it is an explicit test parameter, not a claimed leak detector.

## Output

The program prints:

- engine path
- every IO tensor name
- tensor mode, memory location, data type, and resolved shape
- bytes allocated for each tensor
- total device memory owned by the lesson buffers
- average enqueue time measured with CUDA events
- for lifecycle mode, completed/failed cycle counts plus before/after host RSS and device-memory use

Example shape report:

```text
Engine: ../06_trtexec_engine/outputs/yolov8n_static_fp32.engine
Tensor buffers:
  - images [input, device, float32] shape=1x3x640x640 bytes=4915200
  - output0 [output, device, float32] shape=1x84x8400 bytes=2822400
Total device bytes: 7737600
Average enqueue time: 2.31 ms
Smoke inference completed successfully.
```

Exact tensor names and timing depend on the exported ONNX graph, TensorRT version, GPU, precision,
and engine build settings.

## Code Notes

`TensorRtPtr<T>` wraps TensorRT interfaces in `std::unique_ptr` so runtime, engine, and context
lifetimes are explicit. The execution context is destroyed before the engine, and the engine before
the runtime, because local variables unwind in reverse construction order.

`DeviceBuffer`, `PinnedHostBuffer`, `CudaStream`, and `CudaEvent` are move-only classes. They can be
stored in vectors or returned from helper functions, but they cannot be accidentally copied.

Tensor addresses are bound by name through `setTensorAddress`. This avoids the fragile mental
mapping between binding index, optimization profile, and tensor name that older TensorRT examples
often expose too early.

The lesson intentionally zero-fills input buffers and only checks that enqueue succeeds. Real image
preprocessing and output postprocessing arrive in later lessons; this lesson keeps the focus on
resource lifetime boundaries.

`run_repeated_lifecycle_test` performs three priming cycles before recording memory. CUDA context,
TensorRT runtime, and host allocator caches may initialize lazily, so including early cycles would confuse one-time
initialization with per-cycle growth. For stronger evidence, run the executable under
`compute-sanitizer --tool memcheck`; host-side AddressSanitizer is also useful but does not diagnose
CUDA allocations.

## Checkpoints

- Run once with the static FP32 engine and once with the static FP16 engine. Compare device buffer
  bytes and enqueue time.
- Temporarily pass a missing `--engine` path and confirm that the error is reported without leaking
  already-created resources.
- Run the dynamic engine without `--input-shape`, then add `--input-shape images:1x3x640x640`.
  Explain why dynamic dimensions must be resolved before buffer allocation.
- In `src/tensorrt_raii.cpp`, find every destructor and explain which external API releases the
  resource it owns.
- Run all failure stages with `--repeat 100`. Confirm that each iteration is counted as an expected
  failure and that the memory gate passes.

Acceptance criteria:

- A C++ executable loads a serialized TensorRT engine from lesson 06.
- TensorRT runtime, engine, and context are managed by smart pointers.
- CUDA stream, event, device buffer, and pinned host buffer ownership is managed by RAII classes.
- The program allocates one buffer per TensorRT IO tensor and binds addresses by tensor name.
- Initialization remains exception-safe if file IO, deserialization, CUDA allocation, or enqueue
  fails.
- Injected setup failures release resources acquired at earlier stages.
- Repeated successful and injected-failure cycles do not exceed the declared host/device memory
  growth tolerance.
- You can explain why this matters for 24x7 camera inference services.
