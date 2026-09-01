# 11 - YOLOv8 TensorRT C++

## Purpose

- Build the main portfolio artifact: end-to-end YOLOv8n TensorRT C++ inference.
- Begin converging lesson code into reusable preprocessing, inference, and postprocessing modules.

Lesson 10 made the Python reference easy to inspect. This lesson turns the same pipeline into the
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

## Prerequisites

- Build a compatible lesson 06 TensorRT engine.
- Use `assets/img.jpeg` as the default input image and the pinned development container for building.

## Deliverables

- Reusable preprocessing, TensorRT runner, postprocessing, and visualization libraries
- `yolov8_trt_cpp` command-line executable
- `yolov8_cpp_tests` focused test target and saved inference outputs

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

```bash
cmake -S 11_yolov8_trt_cpp -B 11_yolov8_trt_cpp/build
cmake --build 11_yolov8_trt_cpp/build
```

## Run

Run with the default lesson 06 static FP32 engine and shared image:

```bash
./11_yolov8_trt_cpp/build/yolov8_trt_cpp \
  --engine 06_trtexec_engine/outputs/yolov8n_static_fp32.engine \
  --image assets/img.jpeg \
  --output-dir 11_yolov8_trt_cpp/outputs
```

Use a different engine or image:

```

<details><summary>Example output (local run)</summary>

```text
Engine: 06_trtexec_engine/outputs/yolov8n_static_fp32.engine
Detections: 6
Last latency ms: preprocess=2.75377, h2d=0.265952, enqueue_host=30.7236, gpu_compute=30.4057, d2h=0.138976, postprocess=0.427954, total=35.0845
JSON report: 11_yolov8_trt_cpp/outputs/detections.json
```
</details>
bash
./11_yolov8_trt_cpp/build/yolov8_trt_cpp \
  --engine 06_trtexec_engine/outputs/yolov8n_static_fp16.engine \
  --image assets/img.jpeg \
  --output-dir 11_yolov8_trt_cpp/outputs
```

Adjust thresholds:

```bash
./11_yolov8_trt_cpp/build/yolov8_trt_cpp \
  --engine 06_trtexec_engine/outputs/yolov8n_static_fp32.engine \
  --image assets/img.jpeg --output-dir 11_yolov8_trt_cpp/outputs \
  --confidence 0.20 --iou 0.50 --max-detections 50
```

Collect steady-state samples in one process:

```bash
./11_yolov8_trt_cpp/build/yolov8_trt_cpp \
  --engine 06_trtexec_engine/outputs/yolov8n_static_fp32.engine \
  --image assets/img.jpeg --output-dir 11_yolov8_trt_cpp/outputs \
  --warmup-iterations 5 --iterations 50
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
NVTX ranges mark warmup/measured iterations and the major pipeline stages for lesson 13.

## Tests

```bash
./11_yolov8_trt_cpp/build/yolov8_cpp_tests
```

The tests cover:

- letterbox geometry for a wide image
- invalid empty image input
- IoU/NMS behavior with overlapping boxes
- mapping boxes back from padded network space to original-image coordinates

## Checkpoints

- Compare `outputs/detections.json` with lesson 10's Python JSON.
- Lower `--confidence` and observe the effect on candidate detections.
- Run the FP32 and FP16 engines from lesson 06 and compare GPU compute time.
- Compare the first warmup sample with measured steady-state `gpu_compute` samples.
- Trace the ownership path in `src/tensorrt_runner.cpp`: runtime, engine, context, stream, events,
  and buffers.
