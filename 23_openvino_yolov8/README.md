# 23 - OpenVINO YOLOv8 on CPU

## Purpose

This elective runs the same lesson 05 YOLOv8n ONNX model on an Intel/CPU-oriented runtime and
compares synchronous and asynchronous OpenVINO execution with lesson 15 TensorRT GPU evidence.

OpenVINO is isolated from the pinned TensorRT container packages in a lesson-local dependency
directory. The course pins `openvino==2025.4.1`; do not replace it with an unrecorded upgrade.

## Prerequisites

- Generate lesson 05's static ONNX model.
- Complete lesson 14's canonical performance collection and lesson 15's evidence validation if a
  GPU-versus-CPU report is required.

## Deliverables

- `run_openvino.py` CPU inference and measurement CLI
- `generate_comparison.py` TensorRT/OpenVINO comparison generator
- Metric tests and a documented local dependency setup

## Setup

The reference environment for lessons 00–20 remains the pinned TensorRT development container.
Do not run `pip install openvino` globally there: pip may replace NumPy, packaging, or other packages
used by TensorRT, ONNX Runtime, and Ultralytics lessons.

Lesson 23 uses an isolated layout:

```text
23_openvino_yolov8/
  requirements.txt  # exact versions committed to Git
  .deps/             # locally installed packages, ignored by Git
  outputs/           # generated measurements, ignored by Git
```

Install or rebuild the isolated dependency directory:

```bash
./23_openvino_yolov8/setup_local_deps.sh
```

This uses pip's `--target` mode because the pinned TensorRT container may not include `ensurepip`.
The setup script removes the old `.deps/` first and installs the exact committed versions. It does
not replace global packages.

Verify both the loaded version and path before benchmarking:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=23_openvino_yolov8/.deps python3 -c \
  'import openvino; print(openvino.__version__); print(openvino.__file__)'
```

The version must start with `2025.4.1`, and the path must be under
`23_openvino_yolov8/.deps/`. Use this lesson-local directory for every command in this lesson so
the benchmark cannot silently import a different user or system package.

Clean the lesson environment with:

```bash
rm -rf 23_openvino_yolov8/.deps 23_openvino_yolov8/outputs
```

## Run

Complete the lesson 05 export and validation first. The comparison step also requires lesson 14's
generated canonical TensorRT evidence at
`14_yolov8_int8_quantization_engineering/outputs/tensorrt10/performance/performance.json`. Follow
lesson 14's reproduction procedure, then run lesson 15's report generator to validate the linked
performance, quality, engine, dataset, and runtime identities before comparing results.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=23_openvino_yolov8/.deps \
  python3 23_openvino_yolov8/run_openvino.py
PYTHONNOUSERSITE=1 PYTHONPATH=23_openvino_yolov8/.deps \
  python3 23_openvino_yolov8/generate_comparison.py
```

`run_openvino.py` performs ten warmups per compiled mode and at least 100 measured requests, records
the CPU model, logical CPU count, and OpenVINO device name, and reports P50/P90/P99,
compiles a `LATENCY` model for synchronous requests, compiles a separate `THROUGHPUT` model for
`AsyncInferQueue`, and checks raw output against lesson 05's ONNX Runtime reference.
The async request pool defaults to OpenVINO's model- and device-specific optimal size. Override it
with `--async-jobs N` only for a documented experiment; `--async-jobs 0` selects automatic sizing.
`generate_comparison.py` reads the canonical machine-readable TensorRT evidence consumed and
validated by checkpoint 15 instead of copying numbers by hand. FP32 and FP16 measurements are
required. INT8 appears only when it passed lesson 14's quality gate and was benchmarked.

OpenVINO's `benchmark_app` is installed with the package and provides a runtime-owned reference:

```

<details><summary>Example output (local run)</summary>

```text
comparison report written to outputs/
```
</details>
bash
PYTHONNOUSERSITE=1 PYTHONPATH=23_openvino_yolov8/.deps \
23_openvino_yolov8/.deps/bin/benchmark_app \
  -m 05_torch_to_onnx/outputs/yolov8n.onnx \
  -d CPU \
  -api async \
  -niter 100
```

### Interpretation

- Synchronous latency describes one request at a time.
- Async response latency measures submission through callback completion and includes request-pool
  backpressure. Async inference latency comes from OpenVINO's `InferRequest.latency` and excludes
  waiting for an available request. Neither is directly interchangeable with sync latency.
- TensorRT GPU and OpenVINO CPU use different hardware, so compare deployment constraints as well
  as milliseconds.
- OpenVINO is relevant for Intel servers, CPU-only edge systems, mixed CPU/GPU capacity planning,
  and organizations that need one runtime across several Intel device classes.

FP16 or INT8 CPU comparisons should only be added when the selected CPU and OpenVINO configuration
actually execute those precisions and detection-quality regression is rerun. Do not infer precision
from model file names or configuration requests alone.

## Outputs

- OpenVINO measurements and the generated CPU/GPU comparison are written under ignored `outputs/`.
- Results remain tied to the recorded CPU, GPU, runtime, model, and synchronization methodology.

## Tests

Run the Python tests from the repository root:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=23_openvino_yolov8/.deps \
python3 -m unittest discover -s 23_openvino_yolov8/tests -v
```

Run `Setup` first. These tests import the pinned lesson-local OpenVINO installation.

## Checkpoints

1. Run the same YOLO ONNX model through OpenVINO on CPU.
2. Measure latency and throughput under a recorded CPU and software environment.
3. Explain when an OpenVINO CPU deployment is preferable to TensorRT GPU deployment.
