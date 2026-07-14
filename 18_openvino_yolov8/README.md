# 18 - OpenVINO YOLOv8 on CPU

This elective runs the same lesson 05 YOLOv8n ONNX model on an Intel/CPU-oriented runtime and
compares synchronous and asynchronous OpenVINO execution with lesson 12a TensorRT GPU evidence.

OpenVINO is isolated from the pinned TensorRT container packages in a lesson-local dependency
directory. The course pins `openvino==2025.4.1`; do not replace it with an unrecorded upgrade.

## Environment

```bash
./18_openvino_yolov8/setup_local_deps.sh
```

This uses pip's `--target` mode because the pinned TensorRT container may not include `ensurepip`.
Dependencies stay under the ignored `.deps/` directory and do not replace global packages.

## Run and Compare

```bash
PYTHONPATH=18_openvino_yolov8/.deps python3 18_openvino_yolov8/run_openvino.py
PYTHONPATH=18_openvino_yolov8/.deps python3 18_openvino_yolov8/generate_comparison.py
```

`run_openvino.py` performs ten warmups and at least 100 measured requests, reports P50/P90/P99,
tests `AsyncInferQueue` with four jobs, and checks the raw output against lesson 05's ONNX Runtime
reference. `generate_comparison.py` reads the machine-readable TensorRT evidence from checkpoint 12a
instead of copying numbers by hand.

OpenVINO's `benchmark_app` is installed with the package and provides a runtime-owned reference:

```bash
PYTHONPATH=18_openvino_yolov8/.deps python3 -m openvino.tools.benchmark.main \
  -m 05_torch_to_onnx/outputs/yolov8n.onnx \
  -d CPU \
  -api async \
  -niter 100
```

## Interpretation

- Synchronous latency describes one request at a time.
- Async throughput describes a saturated CPU and includes queueing behavior; it is not directly
  interchangeable with single-request latency.
- TensorRT GPU and OpenVINO CPU use different hardware, so compare deployment constraints as well
  as milliseconds.
- OpenVINO is relevant for Intel servers, CPU-only edge systems, mixed CPU/GPU capacity planning,
  and organizations that need one runtime across several Intel device classes.

FP16 or INT8 CPU comparisons should only be added when the selected CPU and OpenVINO configuration
actually execute those precisions and detection-quality regression is rerun. Do not infer precision
from model file names or configuration requests alone.
