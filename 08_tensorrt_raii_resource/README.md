# 08 - TensorRT RAII Resource

## Purpose

- Make TensorRT C++ code exception-safe and long-running-service friendly.
- Industrial camera systems and edge inference services often run 24x7.
- A small host memory leak, CUDA memory leak, or forgotten TensorRT object can become a production incident.
- RAII proves that resources are released even when early returns or exceptions happen.

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

## Prerequisites

Generate at least one engine with lesson 06:

```bash
python3 06_trtexec_engine/build_and_benchmark.py --builds static_fp32
```

The default executable path expects:

```text
06_trtexec_engine/outputs/yolov8n_static_fp32.engine
```

Serialized TensorRT engines are machine-local artifacts. Rebuild them inside the target container
and on the target GPU/software stack.

## Deliverables

- `tensorrt_raii_lib` reusable C++ library
- `tensorrt_raii_resource` executable
- `tensorrt_raii_config_tests` focused configuration and failure-path tests

## Directory Layout

- `CMakeLists.txt`: target-based C++17 build file that links TensorRT and CUDA Runtime.
- `include/tensorrt_raii.hpp`: small public API for configuring and running the smoke inference.
- `src/tensorrt_raii.cpp`: TensorRT logger, smart-pointer deleter, CUDA RAII wrappers, tensor buffer
  allocation with TensorRT 10 data-type and vectorization checks, address binding, and enqueue timing.
- `src/main.cpp`: command-line parsing and concise reporting.
- `tests/config_tests.cpp`: focused validation of failure-stage configuration.

The reusable logic is built as `tensorrt_raii_lib`, with a thin executable named
`tensorrt_raii_resource` on top.

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

## Build

```bash
cmake -S 08_tensorrt_raii_resource -B 08_tensorrt_raii_resource/build
cmake --build 08_tensorrt_raii_resource/build
ctest --test-dir 08_tensorrt_raii_resource/build --output-on-failure
```

## Run

Run the default static FP32 engine from lesson 06:

```bash
./08_tensorrt_raii_resource/build/tensorrt_raii_resource \
  --engine 06_trtexec_engine/outputs/yolov8n_static_fp32.engine
```

Use a specific engine:

```bash
./08_tensorrt_raii_resource/build/tensorrt_raii_resource \
  --engine 06_trtexec_engine/outputs/yolov8n_static_fp16.engine
```

Run a dynamic engine by supplying runtime input dimensions:

```bash
./08_tensorrt_raii_resource/build/tensorrt_raii_resource \
  --engine 06_trtexec_engine/outputs/yolov8n_dynamic_fp16.engine \
  --input-shape images:1x3x640x640
```

Adjust warmup and measurement count:

```bash
./08_tensorrt_raii_resource/build/tensorrt_raii_resource --warmup 3 --iterations 10
```

Exercise repeated construction/destruction after three unmeasured priming cycles:

```bash
./08_tensorrt_raii_resource/build/tensorrt_raii_resource --repeat 100
```

Inject a failure after the first tensor buffer has been acquired, repeat it, and verify that already
owned resources are released during stack unwinding:

```bash
./08_tensorrt_raii_resource/build/tensorrt_raii_resource \
  --repeat 100 \
  --inject-failure first-buffer \
  --memory-tolerance-mib 16
```

Valid stages are `engine-read`, `runtime`, `engine`, `context`, `first-buffer`, `stream`, and
`enqueue`. A failed memory-stability gate returns a nonzero status. The tolerance accounts for host
allocator and driver caching; it is an explicit test parameter, not a claimed leak detector.

## Outputs

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

## Tests

Run the configured CTest suite:

```bash
ctest --test-dir 08_tensorrt_raii_resource/build --output-on-failure
```

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
