# 10a - End-to-End Validation Report

## Purpose

This checkpoint turns lessons 05, 06a, and 10 into one reviewable report. It deliberately keeps the
scope narrow: one controlled image, raw-output alignment, a C++ smoke test, ownership notes, and an
unoptimized latency baseline.

It does not measure dataset mAP, repeated-run performance, throughput, FP16/INT8 acceptance, or
pipeline stability. Those are later lessons.

## Prerequisites

The commands below use the same image (`assets/img.jpeg`), static FP32 ONNX model/engine pair, and
default detection thresholds.

Run every command from the repository root:

```bash
mkdir -p 10a_end_to_end_validation_report/outputs/cpp
bash 00_environment_check/check_env.sh \
  > 10a_end_to_end_validation_report/outputs/environment_check.log 2>&1

python3 05_torch_to_onnx/export_yolov8_onnx.py
python3 05_torch_to_onnx/validate_onnx_runtime.py \
  --image assets/img.jpeg

python3 06_trtexec_engine/build_and_benchmark.py \
  --builds static_fp32

python3 06a_polygraphy_precision_alignment/align_precision.py \
  --input-npy 05_torch_to_onnx/outputs/input_nchw_float32.npy \
  --engine 06_trtexec_engine/outputs/yolov8n_static_fp32.engine

cmake -S 10_yolov8_trt_cpp -B 10_yolov8_trt_cpp/build
cmake --build 10_yolov8_trt_cpp/build
./10_yolov8_trt_cpp/build/yolov8_cpp_tests \
  > 10a_end_to_end_validation_report/outputs/cpp_tests.log 2>&1

./10_yolov8_trt_cpp/build/yolov8_trt_cpp \
  --engine 06_trtexec_engine/outputs/yolov8n_static_fp32.engine \
  --image assets/img.jpeg \
  --warmup-iterations 10 \
  --iterations 100 \
  --output-dir 10a_end_to_end_validation_report/outputs/cpp
```

The environment command must pass. The precision command must run TensorRT; do not pass
`--skip-trt` for this checkpoint.

The C++ command records ten warmup samples and 100 measured samples. 10a reports the arithmetic
mean for each stage; the raw samples remain in the C++ JSON. Lesson 11 owns timeline analysis.

## Deliverables

- `generate_report.py` evidence validator and report generator
- `outputs/evidence.json` machine-readable evidence
- `reports/10a_end_to_end_validation.md` generated checkpoint report

## Evidence Flow

```text
assets/img.jpeg
  -> lesson 05: PyTorch vs ONNX Runtime raw output
  -> lesson 06a: ONNX Runtime vs TensorRT raw output using lesson 05's saved tensor
  -> lesson 10: C++ end-to-end image, JSON result, and focused tests
  -> generate_report.py: reports/10a_end_to_end_validation.md
```

## Generate the Report

```bash
python3 10a_end_to_end_validation_report/generate_report.py
```

This writes:

- `10a_end_to_end_validation_report/outputs/evidence.json`: selected machine-readable evidence.
- `reports/10a_end_to_end_validation.md`: the reviewable checkpoint report.

The generator rejects mixed images, ONNX models, engines, missing TensorRT alignment, and a missing
passing C++ test log. That prevents a report from combining artifacts from different experiments.

## Outputs

- The runnable commands above produce the files and console evidence described in `Deliverables`.
- Generated build and runtime artifacts remain in the lesson's ignored build or output directory.

## Checkpoints

- Inspect the generated report and explain why raw-output alignment precedes decode and NMS.
- Trace ownership in `10_yolov8_trt_cpp/src/tensorrt_runner.cpp`.
- Explain why the recorded latency is a baseline rather than a performance claim.
- Practice the English summary and walkthrough from the final two report sections.
