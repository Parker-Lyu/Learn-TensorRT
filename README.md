# Learn TensorRT: Production-Oriented C++ Inference & Optimization

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.14.1-green.svg)](https://developer.nvidia.com/tensorrt)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Container](https://img.shields.io/badge/NGC_PyTorch-25.11-orange.svg)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

A practical, hands-on C++17 course focused on **TensorRT 10.14 high-performance deployment** and
**low-latency pipeline engineering**.

Built around an end-to-end YOLOv8 pipeline, this project turns model export into a disciplined
deployment system. The course develops the C++17 foundations behind reliable inference: RAII-based
ownership, explicit resource lifetimes, target-based CMake, defensive validation, reproducible
build and runtime evidence, and interfaces that can evolve from a single executable into reusable
pipeline components.

From there, the curriculum follows the performance-critical path end to end: ONNX graph inspection
and repair, TensorRT engine construction and dynamic shapes, CUDA stream and event synchronization,
measured host-to-device memory paths, NPP/CUDA preprocessing, YOLOv8 postprocessing, explicit Q/DQ
INT8 quantization, bounded asynchronous producer-consumer and multi-stream scheduling, sanitizer
and soak-test evidence, and Nsight Systems/Compute profiling. Advanced electives extend the same
engineering model to TensorRT plugins, Triton, DeepStream, Jetson DLA, OpenVINO, Python bindings,
and LLM inference awareness. The result is not a collection of `trtexec` commands, but a portfolio
of runnable, measurable deployment artifacts with clear correctness, ownership, concurrency, and
performance boundaries.


## Target Audience

This repository is designed for developers bridging the gap between algorithm research and production engineering:

* **Algorithm Engineers Transitioning to Deployment:** You have solid deep learning knowledge (PyTorch, ONNX) but need to master C++/TensorRT for production-grade inference.
* **Junior/Mid-level Deployment Engineers:** You have used the Python TensorRT API or `trtexec`, and want to level up to explicit Q/DQ quantization, CUDA zero-copy, multi-stream asynchronous execution, and Nsight profiling.
* **Engineers Building a Production Portfolio:** You are targeting senior-level Edge CV or high-performance AI deployment roles and need a reference project that demonstrates real-world pipeline architecture.

**Prerequisites:** This is a substantial C++ engineering course. You should have basic familiarity
with ISO C++17 and CMake. Python-only developers should first learn C++ ownership, concurrency, CUDA
execution, and build-system fundamentals.


## Learning Roadmap

You could read [`docs/learning_roadmap.md`](docs/learning_roadmap.md) first. Use
[`docs/coverage_matrix.md`](docs/coverage_matrix.md) to map the core and elective topics to
production deployment skills. Alternatively, you can simply work through the course in order.


The project is organized as small lessons. Each lesson should produce one runnable artifact and one short note about what was learned.
Shared images and other reusable resources live in the root `assets` folder.

## How To Use
This course is developed and tested on Ubuntu. Windows or WSL setups may work, but they are not the reference environment and may require extra troubleshooting. 

1. Clone this repository on an Ubuntu machine with an available NVIDIA GPU:

```bash
git clone git@github.com:Parker-Lyu/Learn-TensorRT.git
cd Learn-TensorRT
```

2. Follow [`00_environment_check/README.md`](00_environment_check/README.md) to start the pinned
development environment.

3. Optionally install VS Code and the Dev Containers extension, then attach to the running container
and open the mounted project directory.

4. Complete the core lessons in order, writing a report at each checkpoint. Then choose electives
from the job descriptions you are targeting instead of completing every elective sequentially.

### Core Path

| Stage | Lessons | Outcome |
| --- | --- | --- |
| Foundation | `00_environment_check` through `11_yolov8_trt_cpp` | End-to-end YOLOv8n TensorRT C++ inference |
| Checkpoint 1 | `12_end_to_end_validation_report` | Reproducible functional and architecture evidence |
| Optimization | `13_nsight_performance_diagnosis`, `14_yolov8_int8_quantization_engineering` | Profiled FP32/FP16/INT8 performance, explicit Q/DQ quantization, and deployment evidence |
| Checkpoint 2 | `15_precision_performance_report` | Application-ready precision and performance report |
| Pipeline | `16_cpp_producer_consumer` through `21_integrated_tensorrt_video_pipeline` | Bounded asynchronous/multi-stream orchestration integrated with CUDA/NPP preprocessing and real TensorRT execution |
| Checkpoint 3 | `22_pipeline_performance_report` | Reproducible load, latency, throughput, sanitizer, restart, and 30-minute soak report; status is generated per environment |


### Elective Tracks

Lessons `23`–`31` use global, stable identifiers. Their numbers do **not** imply that every
elective should be completed in numerical order. Follow the track-specific sequence and
prerequisites below; Lesson `32` is the final synthesis.

| Track | Lessons | Recommended order and prerequisites |
| --- | --- | --- |
| Server inference | `24_triton_inference_server`, `29_cpp_shared_library_python_binding` | Both build on Lesson 17; they are independent of each other. |
| Edge CV | `27_deepstream_gstreamer_multistream`, `28_jetson_orin_xavier_dla_deployment` | Learn 27 before 28 when possible; 28 requires Jetson/DLA hardware for full validation. |
| Advanced TensorRT | `25_onnx_graph_surgery_plugin`, `26_custom_tensorrt_plugin` | **25 → 26**: repair the same unsupported `com.acme::AcmeSwish` model with GraphSurgeon, then with a TensorRT plugin. |
| CPU and Intel | `23_openvino_yolov8` | Independent branch; reuse the ONNX and comparison evidence from Lessons 05/15 as documented. |
| LLM awareness | `30_llm_inference_intro` | Independent branch; no YOLO/TensorRT elective prerequisite. |
| CUDA kernel optimization | `31_nsight_compute_kernel_analysis` | Complete Lesson 13 and understand Lesson 26's custom-kernel boundary; profile a small MLP, optimize LayerNorm, and make operator/network/production decisions. |
| Final synthesis | `32_final_portfolio_case_study` | Complete after the core checkpoints and only the elective evidence you plan to present. |


## General Build Pattern

For each C++ lesson:

```bash
cd <lesson_folder>
cmake -S . -B build
cmake --build build
./build/<executable_name>
```

## Requirements

- An Ubuntu host with a suitable NVIDIA GPU, capable of running Docker and NVIDIA containers.


## 🚀 From Learning to Landing

[![Career Opportunities](https://img.shields.io/badge/Career-Land_Your_Dream_Role-brightgreen.svg?style=for-the-badge&logo=rocket)](https://github.com/Parker-Lyu/Learn-TensorRT)

Happy coding, and may this repository help you land your ideal role in high-performance AI engineering!🎉
