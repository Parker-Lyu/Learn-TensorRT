# Learn TensorRT: Production-Grade C++ Inference & Optimization

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.14.1-green.svg)](https://developer.nvidia.com/tensorrt)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Container](https://img.shields.io/badge/NGC_PyTorch-25.11-orange.svg)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

A practical, zero-fluff C++17 learning repository focused on **TensorRT 10.x High-Performance Deployment** and **Low-Latency Pipeline Engineering**. 

Built around an end-to-end YOLOv8 object detection pipeline, this project bridges the gap between basic model export and senior-level production deployment—covering explicit Q/DQ quantization, CUDA zero-copy preprocessing, asynchronous multi-stream execution, and Nsight Systems profiling.  
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
Shared images and other reusable resources live in the root `assets` folder.

## How To Use
This course is developed and tested on Ubuntu. Windows or WSL setups may work, but they are not the reference environment and may require extra troubleshooting. 

1. Clone this repository on an Ubuntu machine:

```bash
git clone git@github.com:Parker-Lyu/Learn-TensorRT.git
cd Learn-TensorRT
```

2. Use a coding agent such as Codex or Claude Code to prepare the development environment by following the agent guide:

```bash
00_environment_check/agent_env_setup.md
```

(option) 3. Install VS Code and the Dev Containers extension, then attach VS Code to the running container and open the mounted project directory inside the container.

4. Complete the core lessons in order, writing a report at each checkpoint. Then choose electives
from the job descriptions you are targeting instead of completing every elective sequentially.

### Core Path

| Stage | Lessons | Outcome |
| --- | --- | --- |
| Foundation | `00_environment_check` through `10_yolov8_trt_cpp` | End-to-end YOLOv8n TensorRT C++ inference |
| Checkpoint 1 | `10a_end_to_end_validation_report` | Reproducible functional and architecture evidence |
| Optimization | `11_nsight_performance_diagnosis`, `12_yolov8_int8_quantization_engineering` | Profiled FP32/FP16/INT8 performance, quantization evolution, and deployment evidence |
| Checkpoint 2 | `12a_precision_performance_report` | Application-ready precision and performance report |
| Pipeline | `13_cpp_producer_consumer` through `17_cuda_preprocess_npp` | Dynamic, asynchronous, multi-stream inference pipeline |
| Checkpoint 3 | `17a_pipeline_performance_report` | Load, latency, throughput, and stability evidence |


### Elective Tracks

| Track | Lessons |
| --- | --- |
| Server inference | `18a_triton_inference_server`, `21_cpp_shared_library_python_binding` |
| Edge CV | `20_deepstream_gstreamer_multistream`, `20a_jetson_orin_xavier_dla_deployment` |
| Advanced TensorRT | `19_onnx_graph_surgery_plugin`, `19a_custom_tensorrt_plugin` |
| CPU and Intel | `18_openvino_yolov8` |
| LLM awareness | `22_llm_inference_intro` |
| Ongoing interview practice | `23_cpp_interview_katas` |
| Final synthesis | `24_final_portfolio_case_study` |

The roadmap is the source of truth for planned lessons. A lesson directory is added only when its
implementation begins; planned lessons do not use README-only placeholder directories.


## General Build Pattern

For each C++ lesson:

```bash
cd <lesson_folder>
cmake -S . -B build
cmake --build build
./build/<executable_name>
```

## Requirements

- Ubuntu host capable of running Docker and NVIDIA containers
