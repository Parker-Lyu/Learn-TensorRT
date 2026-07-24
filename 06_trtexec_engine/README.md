# 06 - trtexec Engine

This lesson builds TensorRT engines from the simplified ONNX models created in lesson 05 and records
benchmark evidence before any TensorRT C++ runtime code is introduced.

Goal: turn validated, simplified ONNX graphs into reproducible TensorRT engine artifacts and
benchmark reports.

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
validated simplified ONNX
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
- `../05_torch_to_onnx/outputs/yolov8n.onnx`: simplified static ONNX model from lesson 05.
- `../05_torch_to_onnx/outputs/yolov8n_dynamic.onnx`: simplified dynamic ONNX model from lesson 05.

## Prerequisites

Complete lesson 00 and lesson 05 first:

```bash
bash 00_environment_check/check_env.sh
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

## Build And Benchmark

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
transfer time, build status, and runtime identity parsed from the generated evidence.

## Checkpoints

- Compare `static_fp32` and `static_fp16` engine size and GPU compute time.
- Open the `*_layers.json` files and find the first convolution layer.
- Explain why end-to-end latency can differ from GPU compute time.
- Re-run the script and observe that `trtexec_timing.cache` can reduce tactic selection work.
- Build the dynamic ONNX engine and explain the difference between `--minShapes`, `--optShapes`,
  `--maxShapes`, and `--shapes`.

Acceptance criteria:

- Static FP32 and FP16 `.engine` files are generated from the simplified static ONNX under
  `outputs/`.
- Full `trtexec` logs and JSON timing/layer/profile artifacts are recorded.
- The manifest records TensorRT, CUDA Toolkit, GPU, driver, and pinned container identity.
- `outputs/benchmark_summary.md` records latency, throughput, GPU compute time, transfer time, and
  engine size.
- You can compare FP32 and FP16 results using measured evidence.
- If the simplified dynamic ONNX model is present, a dynamic-profile FP16 engine is generated and
  benchmarked.

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

Treat serialized TensorRT engines as machine-local artifacts. If you move from an RTX 2060 laptop to
an RTX 4090 desktop, keep the ONNX and rebuild the `.engine` files on the target machine so tactics
match that GPU and software stack.

## Review: Generated trtexec Commands

`build_and_benchmark.py` 默认会生成下面这些 `trtexec` 命令。路径按从仓库根目录运行时的相对路径展示，实际脚本会解析成绝对路径。

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

这些命令的目的都是把 lesson 05 生成的 ONNX 模型交给 TensorRT 解析、优化、序列化成 `.engine` 文件，同时记录后续复盘需要的 benchmark 和 profiling 证据。`static_fp32` 用作基线，`static_fp16` 用来观察半精度加速和 engine 大小变化，`dynamic_fp16` 用来学习动态输入尺寸下 optimization profile 的配置方式。

关键参数含义：

- `--onnx`: 输入 ONNX 模型路径。
- `--saveEngine`: 输出 TensorRT 序列化 engine 路径。
- `--memPoolSize=workspace:2048`: 设置 TensorRT 构建时可用的 workspace 显存池，单位是 MiB。
- `--timingCacheFile`: 复用 tactic timing cache，让同一环境下重复构建更快、更稳定。
- `--profilingVerbosity=detailed`: 保存更详细的 layer 信息，方便分析 TensorRT 如何优化网络。
- `--dumpLayerInfo` / `--exportLayerInfo`: 打印并导出 TensorRT layer 结构。
- `--dumpProfile` / `--exportProfile`: 打印并导出逐层耗时数据。
- `--separateProfileRun`: 将 profiling run 和主 benchmark run 分开，减少 profiling 对总体延迟统计的影响。
- `--exportTimes`: 导出 benchmark timing samples，供 `summarize_results.py` 汇总。
- `--warmUp`: 正式计时前预热，减少初始化、缓存和 GPU 频率波动带来的噪声。
- `--duration`: benchmark 采样持续时间，时间越长通常统计越稳定。
- `--avgRuns`: 每个 timing sample 内平均的 inference 次数，用于平滑短时抖动。
- `--percentile=50,90,95,99`: 输出 P50、P90、P95、P99 延迟，便于同时观察典型延迟和尾延迟。
- `--noTF32`: 为严格 FP32 对齐基线关闭 TensorRT 默认启用的 TF32。
- `--fp16`: 允许 TensorRT 使用 FP16 tactics 和 FP16 layer，前提是硬件和 layer 支持。
- `--minShapes` / `--optShapes` / `--maxShapes`: 动态 shape engine 的最小、最优、最大输入范围。
- `--shapes`: 本次 benchmark 实际运行的输入 shape，必须落在 profile 范围内。
