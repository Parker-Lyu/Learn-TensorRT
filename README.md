# Learn TensorRT

A focused C++17 learning repository for TensorRT deployment and inference optimization.

The main learning path uses YOLOv8n as the running model, because it is small enough for fast experiments and realistic enough to cover model export, TensorRT engine building, preprocessing, postprocessing, FP16, INT8, async inference, and benchmark reporting.

## Learning Roadmap

Read the full roadmap first:

```bash
docs/learning_roadmap.md
```

For senior production deployment coverage, also read:

```bash
docs/coverage_matrix.md
```

The project is organized as small lessons. Each lesson should produce one runnable artifact and one short note about what was learned.

| Stage | Folder | Goal |
| --- | --- | --- |
| 0 | `00_environment_check` | Verify CUDA, TensorRT, OpenCV, CMake, compiler, and GPU state. |
| 1 | `01_hello_world` | Build a minimal C++17 project with CMake. |
| 1 | `02_opencv_read_show_image` | Load and inspect images with OpenCV. |
| 1 | `03_opencv_preprocess` | Implement resize, letterbox, normalize, HWC-to-CHW, and batch buffer preparation. |
| 2 | `04_cuda_memory_stream` | Learn CUDA memory allocation, host/device copies, and streams. |
| 2 | `05_torch_to_onnx` | Export YOLOv8n from PyTorch to ONNX and validate the graph. |
| 3 | `06_trtexec_engine` | Build and benchmark TensorRT engines with `trtexec`. |
| 3 | `07_tensorrt_raii_resource` | Manage TensorRT and CUDA resources with RAII and smart pointers. |
| 3 | `08_tensorrt_cpp_basic` | Write minimal TensorRT C++ engine loading and inference code. |
| 4 | `09_yolov8_trt_python` | Run YOLOv8n TensorRT inference in Python for fast debugging. |
| 4 | `10_yolov8_trt_cpp` | Implement end-to-end YOLOv8n TensorRT C++ inference. |
| 5 | `11_nsight_performance_diagnosis` | Diagnose CPU/GPU bottlenecks with Nsight Systems and timeline evidence. |
| 5 | `12_yolov8_int8_calibration` | Build INT8 engines and compare accuracy/performance with FP16. |
| 6 | `13_cpp_producer_consumer` | Build a thread-safe producer-consumer image pipeline. |
| 6 | `14_dynamic_batching` | Run TensorRT dynamic batch inference and handle batched buffers. |
| 6 | `15_async_video_pipeline` | Run video or multi-image async inference with preprocessing and postprocessing overlap. |
| 6 | `16_multistream_video_pipeline` | Build a multi-camera or multi-video inference pipeline with scheduling and batching. |
| 7 | `17_cuda_preprocess_npp` | Move preprocessing hotspots to CUDA kernels or NVIDIA NPP. |
| 8 | `18_openvino_yolov8` | Deploy the same ONNX model with OpenVINO for CPU-focused comparison. |
| 9 | `19_onnx_graph_surgery_plugin` | Handle unsupported operators with graph surgery and TensorRT plugins. |
| 10 | `20_deepstream_gstreamer_multistream` | Run production-style multi-stream inference with DeepStream and GStreamer. |
| 10 | `21_cpp_shared_library_python_binding` | Package C++ inference as a shared library and call it from Python. |
| 11 | `22_llm_inference_intro` | Learn the entry-level LLM inference concepts relevant to deployment roles. |
| 11 | `23_cpp_interview_katas` | Practice C++ hand-written deployment kernels and data structures. |
| 12 | `24_benchmark_report` | Produce a clean benchmark report suitable for interviews. |

## Current Lessons

### 01 - Hello World

```bash
cd 01_hello_world
cmake -S . -B build
cmake --build build
./build/hello_world
```

### 02 - OpenCV Read And Show Image

```bash
cd 02_opencv_read_show_image
cmake -S . -B build
cmake --build build
./build/opencv_read_show_image
```

## General Build Pattern

For each C++ lesson:

```bash
cd <lesson_folder>
cmake -S . -B build
cmake --build build
./build/<executable_name>
```

## Requirements

- Ubuntu 22.04 or another recent Linux distribution
- NVIDIA GPU and driver
- CUDA Toolkit
- TensorRT
- C++17 compiler
- CMake 3.10 or newer
- OpenCV
- Python 3 for model export and validation
