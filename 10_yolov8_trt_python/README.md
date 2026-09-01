# 10 - YOLOv8 TensorRT Python

## Purpose

- Build a fast debugging reference before the full C++ implementation.

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

`cuda-python==13.0.3` is pinned by `docker/Dockerfile.dev` to match the CUDA 13.0 course baseline.

The default command expects:

```text
06_trtexec_engine/outputs/yolov8n_static_fp32.engine
assets/img.jpeg
```

## Deliverables

- `infer_yolov8_trt.py` inference CLI
- Saved detection JSON and annotated image under `outputs/`
- Documented TensorRT engine prerequisite

## Run

Run the default FP32 engine on the shared sample image (from the lesson directory so relative defaults resolve):

```bash
cd 10_yolov8_trt_python && python3 infer_yolov8_trt.py
```

<details><summary>Example output (local run, partial)</summary>

```text
Engine: ../06_trtexec_engine/outputs/yolov8n_static_fp32.engine
Image: ../assets/img.jpeg
Input: images (1, 3, 640, 640)
Output: output0 (1, 84, 8400)
Detections: 6
        person 0.898 box=[191.9481658935547, 419.8030700683594, 453.68359375, 1065.5908203125]
          bird 0.811 box=[481.1155700683594, 852.3627319335938, 582.0276489257812, 1067.0]
Latency ms: preprocess=12.95, inference=33.61, postprocess=1.75, total=48.31
Output image: outputs/img_yolov8_trt_python.jpg
Detections JSON: outputs/detections.json
```
</details>

Use a specific engine or image:

```bash
cd 10_yolov8_trt_python && python3 infer_yolov8_trt.py \
  --engine ../06_trtexec_engine/outputs/yolov8n_static_fp16.engine \
  --image ../assets/img.jpeg
```

Run a dynamic engine by supplying runtime dimensions:

```bash
cd 10_yolov8_trt_python && python3 infer_yolov8_trt.py \
  --engine ../06_trtexec_engine/outputs/yolov8n_dynamic_fp16.engine \
  --input-shape images:1x3x640x640
```

Run an optional Ultralytics reference check:

```bash
cd 10_yolov8_trt_python && python3 infer_yolov8_trt.py --reference
```

## Outputs

Generated files are written to `outputs/`:

- `img_yolov8_trt_python.jpg`: input image with detections drawn.
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
