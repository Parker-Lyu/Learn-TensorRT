# 06 - trtexec Engine

This lesson builds TensorRT engines from the ONNX model created in lesson 05 and records benchmark
evidence before any TensorRT C++ runtime code is introduced.

Goal: turn a validated ONNX graph into reproducible TensorRT engine artifacts and benchmark reports.

Topics:

- `trtexec --onnx`
- FP32 and FP16 engine builds
- Static shape engines
- Dynamic shape optimization profiles
- Workspace memory pool
- Layer profiling
- Engine serialization
- Timing cache reuse

## Why This Matters

`trtexec` is the quickest honest boundary test between ONNX export and TensorRT runtime integration.
It answers three questions before C++ code adds more moving parts:

```text
validated ONNX
  -> parse and build TensorRT engine
  -> benchmark generated engine
  -> inspect layer/profile evidence
  -> save serialized engine for later runtime lessons
```

If `trtexec` cannot parse or benchmark the model, a custom C++ loader will not fix the problem.
This lesson also creates engine files that later lessons can load with TensorRT C++ APIs.

## Directory Layout

- `build_and_benchmark.py`: builds FP32 and FP16 engines with `trtexec` and writes logs/JSON files.
- `summarize_results.py`: converts `trtexec` logs into a compact Markdown benchmark summary.
- `outputs/`: generated `.engine`, log, timing, layer, profile, manifest, and summary files. This
  folder is ignored by git.
- `../05_torch_to_onnx/outputs/yolov8n.onnx`: static ONNX model from lesson 05.
- `../05_torch_to_onnx/outputs/yolov8n_dynamic.onnx`: optional dynamic ONNX model from lesson 05.

## Prerequisites

Complete lesson 00 and lesson 05 first:

```bash
bash 00_environment_check/check_env.sh
python3 05_torch_to_onnx/export_yolov8_onnx.py
python3 05_torch_to_onnx/inspect_onnx.py
python3 05_torch_to_onnx/validate_onnx_runtime.py
```

Optional dynamic ONNX export:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py \
  --dynamic \
  --output 05_torch_to_onnx/outputs/yolov8n_dynamic.onnx
```

The dynamic engine build is skipped if the dynamic ONNX file is not present.

## Build And Benchmark

Run from the repository root:

```bash
python3 06_trtexec_engine/build_and_benchmark.py
```

The default command builds:

- `outputs/yolov8n_static_fp32.engine`
- `outputs/yolov8n_static_fp16.engine`
- `outputs/yolov8n_dynamic_fp16.engine`, only when the dynamic ONNX file exists

Each build also writes:

- `*_times.json`: end-to-end timing samples exported by `trtexec`.
- `*_layers.json`: TensorRT layer information.
- `*_profile.json`: per-layer profiling output.
- `*.log`: full `trtexec` console output.
- `build_manifest.json`: paths and settings used by this run.

Preview the exact `trtexec` commands without building engines:

```bash
python3 06_trtexec_engine/build_and_benchmark.py --dry-run
```

Run a focused smoke build when you only want to verify one path:

```bash
python3 06_trtexec_engine/build_and_benchmark.py \
  --builds static_fp32 \
  --warmup-ms 100 \
  --duration-sec 1
```

Tune benchmark duration or workspace:

```bash
python3 06_trtexec_engine/build_and_benchmark.py \
  --workspace-mib 4096 \
  --warmup-ms 1000 \
  --duration-sec 10
```

Build only static engines:

```bash
python3 06_trtexec_engine/build_and_benchmark.py --skip-dynamic
```

Build a specific subset:

```bash
python3 06_trtexec_engine/build_and_benchmark.py --builds static_fp32 static_fp16
```

First-time engine builds can take several minutes because TensorRT selects tactics and writes a
timing cache. Later runs on the same machine can be faster because `outputs/trtexec_timing.cache`
is reused.

## Dynamic Shape Profile

The default dynamic profile assumes YOLOv8 input tensor name `images`:

```text
min: images:1x3x320x320
opt: images:1x3x640x640
max: images:4x3x640x640
run: images:1x3x640x640
```

Override these values if the inspected ONNX tensor name or allowed image sizes are different:

```bash
python3 06_trtexec_engine/build_and_benchmark.py \
  --input-name images \
  --dynamic-min 1x3x320x320 \
  --dynamic-opt 1x3x640x640 \
  --dynamic-max 4x3x640x640
```

The benchmark result is for the shape passed to `--shapes`. The profile range only describes the
shapes the engine is allowed to run.

## Summarize

Create a Markdown benchmark report:

```bash
python3 06_trtexec_engine/summarize_results.py
```

The report is written to:

```text
06_trtexec_engine/outputs/benchmark_summary.md
```

The summary table includes engine size, throughput, end-to-end latency, GPU compute time, H2D/D2H
transfer time, and build status parsed from the logs.

## Checkpoints

- Compare `static_fp32` and `static_fp16` engine size and GPU compute time.
- Open the `*_layers.json` files and find the first convolution layer.
- Explain why end-to-end latency can differ from GPU compute time.
- Re-run the script and observe that `trtexec_timing.cache` can reduce tactic selection work.
- Build the dynamic ONNX engine and explain the difference between `--minShapes`, `--optShapes`,
  `--maxShapes`, and `--shapes`.

Acceptance criteria:

- Static FP32 and FP16 `.engine` files are generated under `outputs/`.
- Full `trtexec` logs and JSON timing/layer/profile artifacts are recorded.
- `outputs/benchmark_summary.md` records latency, throughput, GPU compute time, transfer time, and
  engine size.
- You can compare FP32 and FP16 results using measured evidence.
- If a dynamic ONNX model is present, a dynamic-profile FP16 engine is generated and benchmarked.
