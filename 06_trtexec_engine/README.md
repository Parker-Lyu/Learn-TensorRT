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

First create both explicit FP16 graphs and their validation reports:

```bash
python3 06_trtexec_engine/prepare_fp16_onnx.py
```

Then preview or build the complete matrix:

```bash
python3 06_trtexec_engine/build_and_benchmark.py --dry-run
python3 06_trtexec_engine/build_and_benchmark.py
python3 06_trtexec_engine/summarize_results.py
```

The matrix is `static_fp32`, `static_fp16_legacy`, `dynamic_fp16_legacy`,
`static_fp16_strong`, and `dynamic_fp16_strong`. Legacy builds consume lesson 05 FP32 ONNX and
pass `--fp16`; they are retained for older production environments and are not the new default.
Strong builds consume the validated AutoCast graphs and pass `--stronglyTyped` (never both flags).
Use `--skip-strongly-typed` only when reproducing an old environment that cannot consume explicit
FP16 ONNX.

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
