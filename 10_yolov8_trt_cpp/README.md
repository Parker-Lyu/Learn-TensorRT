# 10 - YOLOv8 TensorRT C++

This lesson builds the first end-to-end C++ YOLOv8 TensorRT deployment artifact.

Goal: accept an image and TensorRT engine, run preprocessing, inference, postprocessing, and
visualization in reusable C++ modules.

Topics:

- OpenCV letterbox preprocessing
- TensorRT runtime deserialization
- Reusable CUDA device and pinned-host buffers, stream, and event timing
- YOLOv8 output decode
- Class-aware NMS
- Coordinate scaling back to the original image
- Visualization
- CLI arguments
- Cold-start and steady-state latency sampling
- Header-only NVTX3 ranges for Nsight Systems (CUDA 13; no removed `nvToolsExt` library)
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
- `src/tensorrt_runner.cpp`: engine loading, TensorRT 10 name-based context setup, reusable CUDA
  buffers, IO-format validation, and enqueue timing.
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
  --image ../assets/img.jpeg \
  --output-dir outputs
```

Adjust thresholds:

```bash
./build/yolov8_trt_cpp --confidence 0.20 --iou 0.50 --max-detections 50
```

Collect steady-state samples in one process:

```bash
./build/yolov8_trt_cpp --warmup-iterations 5 --iterations 50
```

The engine, execution context, CUDA stream, events, and device buffers are created once and reused.
Warmup samples are saved separately and excluded from the measured sample set.

## Outputs

Generated files go to `outputs/`:

- `img_yolov8_trt_cpp.jpg`: annotated image.
- `detections.json`: detections, warmup samples, measured samples, and the final latency breakdown.

The latency fields distinguish:

- `enqueue_host`: CPU time spent calling `enqueueV3`.
- `gpu_compute`: CUDA-event time for TensorRT work on the stream.
- `h2d` and `d2h`: CUDA-event transfer times.
- `total`: preprocessing through postprocessing; engine loading, image decoding, visualization, and
  file writing are excluded.

`enqueue_host` and `gpu_compute` describe different timelines and should not be added together.
NVTX ranges mark warmup/measured iterations and the major pipeline stages for lesson 11.

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
- Run the FP32 and FP16 engines from lesson 06 and compare GPU compute time.
- Compare the first warmup sample with measured steady-state `gpu_compute` samples.
- Trace the ownership path in `src/tensorrt_runner.cpp`: runtime, engine, context, stream, events,
  and buffers.

Acceptance criteria:

- The program accepts image and engine paths.
- It saves an output image with detection boxes.
- It uses `getNbIOTensors`, `getIOTensorName`, `setTensorAddress`, and `enqueueV3` rather than
  deprecated binding-index APIs.
- It reports host enqueue, GPU compute, transfers, preprocessing, postprocessing, and total latency.
- It can separate first-inference behavior from repeated steady-state samples.
- Reusable preprocessing, inference, postprocessing, and visualization code is not trapped inside
  `main`.
- Focused tests cover representative invalid input and boundary cases.
