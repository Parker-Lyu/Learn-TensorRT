# 10 - YOLOv8 TensorRT C++

This lesson builds the first end-to-end C++ YOLOv8 TensorRT deployment artifact.

Goal: accept an image and TensorRT engine, run preprocessing, inference, postprocessing, and
visualization in reusable C++ modules.

Topics:

- OpenCV letterbox preprocessing
- TensorRT runtime deserialization
- CUDA device buffers, stream, and event timing
- YOLOv8 output decode
- Class-aware NMS
- Coordinate scaling back to the original image
- Visualization
- CLI arguments
- Latency breakdown
- Library targets for reusable preprocessing, inference, postprocessing, and visualization
- Focused tests for preprocessing and postprocessing edge cases

## Why This Matters

Lesson 09 made the Python reference easy to inspect. This lesson turns the same pipeline into the
C++ shape used by deployment services:

```text
image
  -> preprocess module
  -> TensorRT runner module
  -> postprocess module
  -> visualization/reporting
```

The code is intentionally split before the program grows into camera streams, batching, async
pipelines, and service-style reporting.

## Directory Layout

- `CMakeLists.txt`: target-based build with separate library targets.
- `include/`: public headers for preprocessing, TensorRT runner, postprocessing, visualization, and
  shared types.
- `src/preprocess.cpp`: OpenCV letterbox and RGB NCHW float32 conversion.
- `src/tensorrt_runner.cpp`: engine loading, TensorRT context setup, CUDA buffers, enqueue timing.
- `src/postprocess.cpp`: YOLOv8 decode, IoU, class-aware NMS, and box mapping.
- `src/visualize.cpp`: detection drawing.
- `src/main.cpp`: CLI, orchestration, image/JSON outputs, latency report.
- `tests/test_preprocess_postprocess.cpp`: focused defensive tests without external test
  dependencies.

## Build

Run from this lesson directory:

```bash
cmake -S . -B build
cmake --build build
```

## Run

Run with the default lesson 06 static FP32 engine and shared image:

```bash
./build/yolov8_trt_cpp
```

Use a different engine or image:

```bash
./build/yolov8_trt_cpp \
  --engine ../06_trtexec_engine/outputs/yolov8n_static_fp16.engine \
  --image ../assets/img2.jpeg \
  --output-dir outputs
```

Adjust thresholds:

```bash
./build/yolov8_trt_cpp --confidence 0.20 --iou 0.50 --max-detections 50
```

## Outputs

Generated files go to `outputs/`:

- `img2_yolov8_trt_cpp.jpg`: annotated image.
- `detections.json`: detections and latency breakdown.

The console report includes preprocessing, H2D copy, TensorRT enqueue, D2H copy, postprocessing, and
total latency.

## Tests

Run focused tests from the lesson build directory:

```bash
./build/yolov8_cpp_tests
```

The tests cover:

- letterbox geometry for a wide image
- invalid empty image input
- IoU/NMS behavior with overlapping boxes
- mapping boxes back from padded network space to original-image coordinates

## Checkpoints

- Compare `outputs/detections.json` with lesson 09's Python JSON.
- Lower `--confidence` and observe the effect on candidate detections.
- Run the FP32 and FP16 engines from lesson 06 and compare enqueue time.
- Trace the ownership path in `src/tensorrt_runner.cpp`: runtime, engine, context, stream, events,
  and buffers.

Acceptance criteria:

- The program accepts image and engine paths.
- It saves an output image with detection boxes.
- It reports preprocessing, inference, postprocessing, and total latency.
- Reusable preprocessing, inference, postprocessing, and visualization code is not trapped inside
  `main`.
- Focused tests cover representative invalid input and boundary cases.
