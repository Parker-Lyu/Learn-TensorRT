# 08 - TensorRT C++ Basic

This lesson writes the first complete TensorRT C++ path: parse ONNX, build a serialized engine,
deserialize it with the runtime API, allocate tensor buffers, bind tensor addresses, and enqueue one
inference.

Goal: understand the minimal TensorRT C++ runtime flow without hiding ownership or build steps.

Topics:

- TensorRT logger
- Builder, network definition, builder config, and workspace memory pool
- ONNX parser through `nvonnxparser`
- TensorRT 10 strongly typed network creation
- Strict FP32 builds by default, with optional TF32 kernel math
- TensorRT timing cache reuse for tactic selection
- Runtime creation and engine deserialization
- Execution context creation
- Tensor names, tensor modes, shapes, data types, and memory locations
- CUDA device buffers, pinned host buffers, stream, and events
- Host-to-device input copy, `enqueueV3`, and device-to-host output copy

## Why This Matters

Lesson 06 used `trtexec` to prove that the ONNX model can become a TensorRT engine. Lesson 07
focused on RAII resource lifetime. This lesson connects both ideas in a compact C++ program:

```text
ONNX file
  -> TensorRT builder + ONNX parser
  -> serialized engine bytes
  -> TensorRT runtime deserialization
  -> execution context
  -> IO tensor buffers
  -> one CUDA stream enqueue
```

The code still uses a synthetic zero input. Real image preprocessing and YOLO postprocessing arrive
in later lessons; here the learning target is the TensorRT API sequence and the ownership model.

## Directory Layout

- `CMakeLists.txt`: target-based C++17 build that links CUDA Runtime, TensorRT, and `nvonnxparser`.
- `include/tensorrt_basic.hpp`: small public config/report API.
- `src/tensorrt_basic.cpp`: engine building, serialization, deserialization, tensor buffer
  allocation, copy/enqueue/copy flow, and output checksum.
- `src/main.cpp`: command-line parsing and console report.
- `outputs/`: generated `.engine` and timing cache files from this lesson. This folder is ignored
  by git.

The reusable logic is built as `tensorrt_cpp_basic_lib`; the runnable artifact is
`tensorrt_cpp_basic`.

## Prerequisites

Generate the static ONNX model from lesson 05:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py
```

For the dynamic-shape experiment, also generate:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py --dynamic
```

## Build

Run from this lesson directory:

```bash
cmake -S . -B build
cmake --build build
```

## Run

Build a static FP32 engine from lesson 05 ONNX, save it under this lesson's `outputs/`, deserialize
it, and run one smoke inference:

```bash
./build/tensorrt_cpp_basic
```

Allow TF32 kernel math while keeping the ONNX and engine tensor types FP32:

```bash
./build/tensorrt_cpp_basic \
  --allow-tf32 \
  --engine outputs/yolov8n_cpp_basic_tf32.engine
```

TensorRT 10.12 deprecated `BuilderFlag::kFP16` in favor of strong typing. This lesson therefore does
not offer a builder-level `--fp16` switch. Precision-changing workflows should encode types in the
model, for example with the ModelOpt explicit-Q/DQ path introduced in lesson 12.

First-time TensorRT builds can take several minutes because tactic selection happens inside the
C++ builder, just like it did through `trtexec`. Strict FP32 builds use
`outputs/tensorrt_timing_fp32.cache`; builds with `--allow-tf32` use
`outputs/tensorrt_timing_tf32.cache`. The builder performs strict timing-cache header verification,
reads the selected cache before tactic selection, and writes it after a successful engine build.
Re-run a build with the same ONNX, shape, math policy, GPU, driver, CUDA, and TensorRT stack to reuse
measured tactic timings. Re-run with `--load-engine` when you want a quick runtime smoke test after
the engine already exists.

Load an engine that this lesson already built, skipping ONNX parsing and build time:

```bash
./build/tensorrt_cpp_basic \
  --load-engine \
  --engine outputs/yolov8n_cpp_basic.engine
```

Build and run the dynamic ONNX model with a single-shape optimization profile:

```bash
./build/tensorrt_cpp_basic \
  --onnx ../05_torch_to_onnx/outputs/yolov8n_dynamic.onnx \
  --engine outputs/yolov8n_dynamic_cpp_basic.engine \
  --timing-cache outputs/tensorrt_timing_dynamic.cache \
  --input-shape images:1x3x640x640
```

Tune workspace and measurement count:

```bash
./build/tensorrt_cpp_basic --workspace-mib 4096 --warmup 3 --iterations 10
```

## Output

The program prints:

- ONNX path, unless `--load-engine` was used
- engine path and serialized engine size
- timing cache path, whether it was loaded or created, and serialized cache size when building
- whether the engine was built in this run
- whether the network was strongly typed and whether TF32 math was allowed (build mode only)
- every IO tensor name, mode, memory location, type, shape, and byte count
- output tensor checksum after device-to-host copy
- total device and pinned host memory owned by the lesson buffers
- average enqueue time measured with CUDA events

Example:

```text
ONNX: ../05_torch_to_onnx/outputs/yolov8n.onnx
Engine: outputs/yolov8n_cpp_basic.engine
Engine source: built from ONNX
Engine bytes: 13215908
Timing cache: outputs/tensorrt_timing_fp32.cache (created, written, bytes=8123456)
Strongly typed network: yes
TF32 allowed: no
Tensor buffers:
  - images [input, device, float32] shape=1x3x640x640 bytes=4915200
  - output0 [output, device, float32] shape=1x84x8400 bytes=2822400 checksum=12735248971051671350
Total device bytes: 7737600
Total pinned host bytes: 7737600
Average enqueue time: 4.27 ms
C++ TensorRT basic flow completed successfully.
```

Exact engine size, timing, and checksum can change with TensorRT version, GPU, tactic selection,
precision mode, and ONNX export settings.

## Code Notes

The build path creates:

```text
IBuilder -> INetworkDefinition -> nvonnxparser::IParser -> IBuilderConfig -> IHostMemory
```

When building from ONNX, the lesson attaches an `ITimingCache` to `IBuilderConfig`. If the cache file
already exists, TensorRT can reuse compatible tactic measurements; after a successful build the cache
is serialized back to disk. Timing caches are machine- and stack-sensitive, so keep them local to the
GPU, driver, CUDA, TensorRT version, model shape, and precision experiment that produced them.

The runtime path then creates:

```text
IRuntime -> ICudaEngine -> IExecutionContext
```

The lesson intentionally deserializes the engine bytes even when the same process just built them.
That keeps the runtime flow identical to later applications that load a prebuilt engine artifact.

Dynamic ONNX inputs need an optimization profile before building and runtime dimensions before
buffer allocation. This lesson uses one supplied shape as min/opt/max to keep the API visible
without introducing profile tuning yet. The builder owns the profile returned by
`createOptimizationProfile`, so the code keeps that ownership boundary explicit.

The network is created with `kSTRONGLY_TYPED`. The default build also clears `BuilderFlag::kTF32` so
the FP32 reference does not silently use TensorFloat-32 math.

## Checkpoints

- Run the default FP32 build, then run `--load-engine` and compare startup time.
- Delete only `outputs/tensorrt_timing_fp32.cache`, rebuild, and compare build time against a
  rebuild that keeps the cache file.
- Build with `--allow-tf32` and compare build time and average enqueue time against strict FP32.
  Explain why both engines still expose FP32 tensors.
- Delete `outputs/yolov8n_cpp_basic.engine` and explain why a fresh build takes longer than loading.
- Run the dynamic ONNX command without `--input-shape`, then add it back and explain the error.
- In `src/tensorrt_basic.cpp`, trace the order in which builder/parser objects, runtime objects, and
  CUDA buffers are created.

Acceptance criteria:

- A C++ executable builds a strongly typed TensorRT 10 engine from ONNX using the ONNX parser.
- The serialized engine is written to `outputs/`.
- A TensorRT timing cache is strictly validated, read before engine build, and written after a
  successful engine build.
- The engine is deserialized through `IRuntime`.
- The program creates an execution context, validates TensorRT 10 IO types/formats, allocates IO
  buffers, binds tensor addresses by name,
  copies input data to the device, enqueues inference, copies outputs back, and reports a checksum.
- Builder, parser, runtime, engine, context, buffer, event, and stream lifetimes are explicit and
  exception-safe.
