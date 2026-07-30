# 06 - trtexec Engine

## Purpose

- Learn TensorRT engine construction before writing C++ code.

`trtexec` is the quickest honest boundary test between ONNX export and TensorRT runtime integration.
It answers three questions before C++ code adds more moving parts:

```text
validated simplified ONNX
  -> parse and build TensorRT engine
  -> benchmark generated engine
  -> inspect layer/profile evidence
  -> save serialized engine for later runtime lessons
```

If `trtexec` cannot parse or benchmark the model, a custom C++ loader will not fix the problem.
This lesson also creates engine files that later lessons can load with TensorRT C++ APIs.

## Prerequisites

Complete lesson 05 first:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py
python3 05_torch_to_onnx/export_yolov8_onnx.py --dynamic
python3 05_torch_to_onnx/inspect_onnx.py
python3 05_torch_to_onnx/validate_onnx_runtime.py
```

The lesson 05 export commands simplify both ONNX graphs by default. Inspect the dynamic graph when
you want explicit evidence for the optimization profile input shape:

```bash
python3 05_torch_to_onnx/inspect_onnx.py \
  --onnx 05_torch_to_onnx/outputs/yolov8n_dynamic.onnx \
  --report 05_torch_to_onnx/outputs/onnx_dynamic_inspection.json
```

The dynamic engine build is skipped if the dynamic ONNX file is not present.
For the normal course path, generate both lesson 05 ONNX files first so lesson 06 builds the static
and dynamic engines from the same simplified model handoff.

## Deliverables

- `build_and_benchmark.py` engine-build and benchmark driver
- `summarize_results.py` evidence summarizer
- Ignored engines, timing cache, logs, profiles, timing samples, manifest, and benchmark summary

## Directory Layout

- `build_and_benchmark.py`: builds FP32 and FP16 engines with `trtexec` and writes logs/JSON files.
- `summarize_results.py`: converts `trtexec` logs into a compact Markdown benchmark summary.
- `outputs/`: generated `.engine`, log, timing, layer, profile, manifest, and summary files. This
  folder is ignored by git.
- `../05_torch_to_onnx/outputs/yolov8n.onnx`: simplified static ONNX model from lesson 05.
- `../05_torch_to_onnx/outputs/yolov8n_dynamic.onnx`: simplified dynamic ONNX model from lesson 05.

## Run

Run from the repository root:

```bash
python3 06_trtexec_engine/build_and_benchmark.py
```

The default command builds:

- `outputs/yolov8n_static_fp32.engine` (strict FP32 reference with TF32 disabled)
- `outputs/yolov8n_static_fp16.engine`
- `outputs/yolov8n_dynamic_fp16.engine`, only when the simplified dynamic ONNX file exists

By default, both static builds consume
`../05_torch_to_onnx/outputs/yolov8n.onnx`, and the dynamic build consumes
`../05_torch_to_onnx/outputs/yolov8n_dynamic.onnx`.

Each build also writes:

- `*_times.json`: end-to-end timing samples exported by `trtexec`.
- `*_layers.json`: TensorRT layer information.
- `*_profile.json`: per-layer profiling output.
- `*.log`: full `trtexec` console output.
- `build_manifest.json`: paths, settings, and the TensorRT/CUDA/GPU/driver/container identity used by this run.

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

### Dynamic Shape Profile

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

### Summarize

Create a Markdown benchmark report:

```bash
python3 06_trtexec_engine/summarize_results.py
```

The report is written to:

```text
06_trtexec_engine/outputs/benchmark_summary.md
```

The summary table includes engine size, throughput, end-to-end latency, GPU compute time, H2D/D2H
transfer time, build status, and runtime identity parsed from the generated evidence.

### Review: Generated trtexec Commands

By default, `build_and_benchmark.py` generates the following `trtexec` commands. The paths below
are shown relative to the repository root; the script resolves them to absolute paths at runtime.

Static FP32 engine:

```bash
trtexec \
  --onnx=05_torch_to_onnx/outputs/yolov8n.onnx \
  --saveEngine=06_trtexec_engine/outputs/yolov8n_static_fp32.engine \
  --memPoolSize=workspace:2048 \
  --timingCacheFile=06_trtexec_engine/outputs/trtexec_timing.cache \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --dumpProfile \
  --separateProfileRun \
  --exportTimes=06_trtexec_engine/outputs/yolov8n_static_fp32_times.json \
  --exportLayerInfo=06_trtexec_engine/outputs/yolov8n_static_fp32_layers.json \
  --exportProfile=06_trtexec_engine/outputs/yolov8n_static_fp32_profile.json \
  --warmUp=500 \
  --duration=5 \
  --avgRuns=10 \
  --percentile=50,90,95,99 \
  --noTF32
```

Static FP16 engine:

```bash
trtexec \
  --onnx=05_torch_to_onnx/outputs/yolov8n.onnx \
  --saveEngine=06_trtexec_engine/outputs/yolov8n_static_fp16.engine \
  --memPoolSize=workspace:2048 \
  --timingCacheFile=06_trtexec_engine/outputs/trtexec_timing.cache \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --dumpProfile \
  --separateProfileRun \
  --exportTimes=06_trtexec_engine/outputs/yolov8n_static_fp16_times.json \
  --exportLayerInfo=06_trtexec_engine/outputs/yolov8n_static_fp16_layers.json \
  --exportProfile=06_trtexec_engine/outputs/yolov8n_static_fp16_profile.json \
  --warmUp=500 \
  --duration=5 \
  --avgRuns=10 \
  --percentile=50,90,95,99 \
  --fp16
```

Dynamic FP16 engine:

```bash
trtexec \
  --onnx=05_torch_to_onnx/outputs/yolov8n_dynamic.onnx \
  --saveEngine=06_trtexec_engine/outputs/yolov8n_dynamic_fp16.engine \
  --memPoolSize=workspace:2048 \
  --timingCacheFile=06_trtexec_engine/outputs/trtexec_timing.cache \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --dumpProfile \
  --separateProfileRun \
  --exportTimes=06_trtexec_engine/outputs/yolov8n_dynamic_fp16_times.json \
  --exportLayerInfo=06_trtexec_engine/outputs/yolov8n_dynamic_fp16_layers.json \
  --exportProfile=06_trtexec_engine/outputs/yolov8n_dynamic_fp16_profile.json \
  --warmUp=500 \
  --duration=5 \
  --avgRuns=10 \
  --percentile=50,90,95,99 \
  --fp16 \
  --minShapes=images:1x3x320x320 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:4x3x640x640 \
  --shapes=images:1x3x640x640
```

Each command asks TensorRT to parse and optimize the lesson 05 ONNX model, serialize an `.engine`
file, and retain the benchmark and profiling evidence needed for later review. `static_fp32` is the
strict reference, `static_fp16` demonstrates half-precision performance and engine-size changes,
and `dynamic_fp16` introduces optimization profiles for dynamic input shapes.

Key arguments:

- `--onnx`: Selects the input ONNX model.
- `--saveEngine`: Selects the serialized TensorRT engine output.
- `--memPoolSize=workspace:2048`: Gives the builder a 2048 MiB workspace memory pool.
- `--timingCacheFile`: Reuses the tactic timing cache to make repeated builds in the same
  environment faster and more consistent.
- `--profilingVerbosity=detailed`: Retains detailed layer metadata for inspecting TensorRT's graph
  optimization decisions.
- `--dumpLayerInfo` / `--exportLayerInfo`: Prints and exports the TensorRT layer structure.
- `--dumpProfile` / `--exportProfile`: Prints and exports per-layer timing data.
- `--separateProfileRun`: Separates layer profiling from the main benchmark so profiling overhead
  does not distort the primary latency samples.
- `--exportTimes`: Exports timing samples consumed by `summarize_results.py`.
- `--warmUp`: Warms up initialization state, caches, and GPU clocks before measurement.
- `--duration`: Sets the benchmark sampling duration; longer runs usually produce more stable
  statistics.
- `--avgRuns`: Averages multiple inferences inside each timing sample to reduce short-lived noise.
- `--percentile=50,90,95,99`: Reports typical and tail latency at P50, P90, P95, and P99.
- `--noTF32`: Disables TensorRT's default TF32 behavior for the strict FP32 alignment baseline.
- `--fp16`: Allows FP16 tactics and layers when the GPU and operation support them.
- `--minShapes` / `--optShapes` / `--maxShapes`: Declare the minimum, preferred, and maximum shapes
  in the dynamic optimization profile.
- `--shapes`: Selects the concrete benchmark shape, which must fall inside the profile range.

## Outputs

- The runnable commands above produce the files and console evidence described in `Deliverables`.
- Generated build and runtime artifacts remain in the lesson's ignored build or output directory.

## Checkpoints

- Compare `static_fp32` and `static_fp16` engine size and GPU compute time.
- Open the `*_layers.json` files and find the first convolution layer.
- Explain why end-to-end latency can differ from GPU compute time.
- Re-run the script and observe that `trtexec_timing.cache` can reduce tactic selection work.
- Build the dynamic ONNX engine and explain the difference between `--minShapes`, `--optShapes`,
  `--maxShapes`, and `--shapes`.


## Appendix: trtexec Arguments

`build_and_benchmark.py` builds each engine by assembling a `trtexec` command. These flags define
the model input, serialized engine output, profiling evidence, benchmark duration, and dynamic-shape
profile.

| Argument | Purpose | Why this lesson uses it |
| --- | --- | --- |
| `trtexec` | TensorRT command-line build and benchmark tool. | Gives a direct ONNX-to-engine boundary test before adding C++ runtime code. |
| `--onnx=<path>` | Input ONNX graph to parse. | Uses the simplified ONNX artifacts from lesson 05. |
| `--saveEngine=<path>` | Writes the serialized TensorRT engine. | Produces `.engine` files that later C++ lessons can load. |
| `--memPoolSize=workspace:<MiB>` | Sets the TensorRT workspace memory pool. | Controls how much temporary GPU memory TensorRT can use while selecting tactics. Larger values can enable faster tactics but use more memory. |
| `--timingCacheFile=<path>` | Reads and writes TensorRT tactic timing data. | Makes repeated builds faster and more reproducible on the same GPU, driver, CUDA, and TensorRT stack. |
| `--profilingVerbosity=detailed` | Stores detailed layer metadata in the engine/profile output. | Makes layer inspection more useful when diagnosing performance. |
| `--dumpLayerInfo` | Prints TensorRT layer information. | Captures the optimized network structure in the log and exported layer file. |
| `--dumpProfile` | Prints per-layer runtime profiling data. | Shows which layers consume time during benchmark runs. |
| `--separateProfileRun` | Runs profiling separately from the main timing loop. | Keeps profiling overhead from distorting the primary benchmark timing. |
| `--exportTimes=<path>` | Writes benchmark timing samples as JSON. | Provides machine-readable latency evidence for summaries and later reports. |
| `--exportLayerInfo=<path>` | Writes layer information as JSON. | Preserves the optimized TensorRT layer inventory for inspection. |
| `--exportProfile=<path>` | Writes per-layer profile results as JSON. | Preserves layer timing data for comparison across precision modes and hardware. |
| `--warmUp=<ms>` | Runs inference before measurement starts. | Reduces first-run noise from lazy initialization, clock ramp-up, and cache effects. |
| `--duration=<sec>` | Sets benchmark measurement time. | Longer runs give more stable latency and throughput numbers. Short runs are useful for smoke tests. |
| `--avgRuns=<n>` | Averages timing over groups of inference runs. | Smooths short-run jitter before reporting each timing sample. |
| `--percentile=50,90,95,99` | Reports selected latency percentiles. | Shows both typical latency and tail latency instead of only an average. |
| `--noTF32` | Disables TF32 for the strict FP32 reference engine. | TensorRT 10 enables TF32 by default; disabling it prevents a supposedly FP32 alignment baseline from silently using TF32 math. |
| `--fp16` | Allows FP16 tactics and FP16 engine layers where supported. | Builds the FP16 comparison engine and usually improves throughput on modern NVIDIA GPUs. |
| `--minShapes=<name:shape>` | Minimum shape allowed by a dynamic optimization profile. | Defines the smallest input shape the dynamic engine must support. |
| `--optShapes=<name:shape>` | Shape TensorRT optimizes most heavily for a dynamic profile. | Tells TensorRT the expected/common shape, usually the main benchmark shape. |
| `--maxShapes=<name:shape>` | Maximum shape allowed by a dynamic optimization profile. | Defines the largest input shape the dynamic engine must support. |
| `--shapes=<name:shape>` | Actual input shape used for this benchmark run. | Measures one concrete shape inside the dynamic profile range. |

Dynamic-shape flags are required only for `dynamic_fp16`. Static engines have fixed input dimensions
from the ONNX graph, so they do not need `--minShapes`, `--optShapes`, `--maxShapes`, or `--shapes`.

Treat serialized TensorRT engines as machine-local artifacts. Keep the ONNX model and rebuild the
`.engine` files on each target machine so tactics match its GPU and software stack.
