# 09 - YOLOv8 TensorRT Python

This lesson builds a Python reference pipeline around the TensorRT engine artifacts from lesson 06.

Goal: run end-to-end YOLOv8 inference in Python so preprocessing, output decode, NMS, and
visualization can be debugged quickly before the C++ deployment version.

Topics:

- TensorRT Python runtime
- CUDA buffers through `cuda-python` Runtime API
- NumPy/OpenCV letterbox preprocessing
- YOLOv8 output decode
- Class-aware NMS
- Coordinate mapping back to the original image
- Visualization and JSON reporting
- Optional Ultralytics reference check

## Why This Matters

Python is the fastest place to inspect model outputs and postprocessing math:

```text
image
  -> letterbox + RGB NCHW float32
  -> TensorRT Python runtime
  -> YOLOv8 decode
  -> NMS
  -> boxes on the original image
```

The C++ lesson that follows should not invent new math. It should port a reference pipeline that is
already easy to inspect.

## Prerequisites

Complete lessons 05 and 06 first:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py
python3 06_trtexec_engine/build_and_benchmark.py --builds static_fp32
```

The default command expects:

```text
06_trtexec_engine/outputs/yolov8n_static_fp32.engine
assets/img2.jpeg
```

## Run

Run from this lesson directory:

```bash
python3 infer_yolov8_trt.py
```

Use a specific engine or image:

```bash
python3 infer_yolov8_trt.py \
  --engine ../06_trtexec_engine/outputs/yolov8n_static_fp16.engine \
  --image ../assets/dog.webp
```

Run a dynamic engine by supplying runtime dimensions:

```bash
python3 infer_yolov8_trt.py \
  --engine ../06_trtexec_engine/outputs/yolov8n_dynamic_fp16.engine \
  --input-shape images:1x3x640x640
```

Run an optional Ultralytics reference check:

```bash
python3 infer_yolov8_trt.py --reference
```

## Outputs

Generated files are written to `outputs/`:

- `img2_yolov8_trt_python.jpg`: input image with detections drawn.
- `detections.json`: tensor metadata, letterbox parameters, latency breakdown, detections, and
  optional Ultralytics top-detection reference.

Example console output:

```text
Input: images (1, 3, 640, 640)
Output: output0 (1, 84, 8400)
Detections: 3
           dog 0.89 box=[...]
Latency ms: preprocess=2.31, inference=4.82, postprocess=1.40, total=8.53
```

Exact detections and latency depend on the engine precision, GPU, TensorRT version, and thresholds.

## Checkpoints

- Open `outputs/detections.json` and confirm the letterbox padding matches lesson 05.
- Lower `--confidence` to `0.05` and observe why NMS is needed.
- Compare static FP32 and static FP16 engines.
- Run `--reference` and compare the top class and approximate box with Ultralytics.

Acceptance criteria:

- The script runs inference on `assets/img2.jpeg` with a TensorRT engine.
- It saves an annotated detection image and JSON report.
- YOLOv8 decode, NMS, and coordinate scaling are implemented in local Python code.
- Device-to-host copies are synchronized before host output buffers are read.
- Optional Ultralytics reference output can be recorded for debugging.
