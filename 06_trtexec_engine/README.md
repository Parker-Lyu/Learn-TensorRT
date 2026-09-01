# 06 - FP16 ONNX and TensorRT engines

## Purpose

This lesson is the precision hand-off between ONNX export and TensorRT C++. It compares the
deprecated weakly typed `trtexec --fp16` compatibility route with the recommended explicit
ModelOpt AutoCast ONNX + TensorRT `--stronglyTyped` route.

## Prerequisites

Run lesson 05 in the persistent development container and keep its validated artifacts:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py
python3 05_torch_to_onnx/export_yolov8_onnx.py --dynamic
python3 05_torch_to_onnx/validate_onnx_runtime.py
```

## Deliverables

- `prepare_fp16_onnx.py`: ModelOpt AutoCast conversion and CPU ONNX Runtime numerical gate.
- `build_and_benchmark.py`: FP32, legacy weakly typed FP16, and strongly typed FP16 engine matrix.
- `summarize_results.py`: benchmark evidence report.
- Ignored ONNX, engines, logs, profiles, timing cache, and JSON reports in `outputs/`.

## Run

Prepare validated explicit-FP16 ONNX graphs (default preparation step):

```bash
python3 06_trtexec_engine/prepare_fp16_onnx.py
```

<details><summary>Example output (local run, partial)</summary>

```text
static: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/yolov8n_static_autocast_fp16.onnx
validation: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/static_fp16_onnx_validation.json
dynamic: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/yolov8n_dynamic_autocast_fp16.onnx
validation: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/dynamic_fp16_onnx_validation.json
```
</details>

Preview the complete five-engine command matrix without executing it:

```bash
python3 06_trtexec_engine/build_and_benchmark.py --dry-run
```

<details><summary>Example output (partial)</summary>

```text
== static_fp32 ==
trtexec --onnx=/workspace/Learn-TensorRT/05_torch_to_onnx/outputs/yolov8n.onnx ... --noTF32
== static_fp16_legacy ==
trtexec ... --fp16
== dynamic_fp16_legacy ==
trtexec ... --fp16 --minShapes=images:1x3x320x320 ...
== static_fp16_strong ==
trtexec ... --stronglyTyped
== dynamic_fp16_strong ==
trtexec ... --stronglyTyped --minShapes=images:1x3x320x320 ...
```
</details>

Build and benchmark the full matrix:

```bash
python3 06_trtexec_engine/build_and_benchmark.py
```

Example output:

```text
manifest: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/build_manifest.json
```

Summarize benchmark artifacts into a Markdown report:

```bash
python3 06_trtexec_engine/summarize_results.py
```

Example output:

```text
report: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/benchmark_summary.md
```

## Outputs

Committed deliverables are the scripts and documentation. Generated files are ignored, including
`yolov8n_*_autocast_fp16.onnx`, `*_fp16_onnx_validation.json`, engines, logs, and benchmark reports.
The manifest records precision, typing mode, deprecation status, source model, and validation report.

## Tests

```bash
python3 -m py_compile 06_trtexec_engine/prepare_fp16_onnx.py 06_trtexec_engine/build_and_benchmark.py
python3 06_trtexec_engine/build_and_benchmark.py --dry-run
```

The first command is CPU-only; engine builds and benchmarks require TensorRT and a GPU.

## Checkpoints

1. Why is `--fp16` called weakly typed, and why is it deprecated for new deployments?
2. Inspect both validation JSON files. Which output channels require a relative tolerance because
   they contain pixel-scale box coordinates?
3. Confirm the strongly typed command contains `--stronglyTyped` and does not contain `--fp16`.
