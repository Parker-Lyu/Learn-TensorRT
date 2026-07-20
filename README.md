# Learn TensorRT

A focused C++17 learning repository for TensorRT deployment and inference optimization.

The main learning path uses YOLOv8n as the running model, because it is small enough for fast experiments and realistic enough to cover model export, TensorRT engine building, preprocessing, postprocessing, FP16, INT8, async inference, accuracy regression checks, and benchmark reporting.

## Learning Roadmap

Read the full roadmap first:

```bash
docs/learning_roadmap.md
```

For senior production deployment coverage, also read:

```bash
docs/coverage_matrix.md
```

For environment setup, Dev Containers usage, and GPU/TensorRT troubleshooting, use the agent-oriented guide in the first lesson:

```bash
00_environment_check/agent_env_setup.md
```

The project is organized as small lessons. Each lesson should produce one runnable artifact and one short note about what was learned.
Shared images and other reusable resources live in the root `assets` folder.

## How To Use

1. Clone this repository on an Ubuntu machine:

```bash
git clone git@github.com:Parker-Lyu/Learn-TensorRT.git
cd Learn-TensorRT
```

2. Use a coding agent such as Codex or Claude Code to prepare the development environment by following the agent guide:

```bash
00_environment_check/agent_env_setup.md
```

This course is developed and tested on Ubuntu. Windows or WSL setups may work, but they are not the reference environment and may require extra troubleshooting.

3. Start the TensorRT container and bind-mount this project directory into the container. Keep the course image unchanged unless a lesson explicitly asks for a separate experiment:

```bash
nvcr.io/nvidia/tensorrt:23.10-py3
```

Lesson 12 explicitly adds a second, version-pinned ModelOpt environment based on
`nvcr.io/nvidia/pytorch:25.11-py3`. TensorRT 8.6 and TensorRT 10 evidence are kept in separate
reference bundles and are never mixed.

4. Install VS Code and the Dev Containers extension, then attach VS Code to the running container and open the mounted project directory inside the container.

5. Complete the core lessons in order, writing a report at each checkpoint. Then choose electives
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

Lesson 12 is an advanced quantization case study rather than a basic calibration demo. Its recorded
result is deliberately evidence-driven: explicit Q/DQ restores INT8 quality, but matched TensorRT
10 measurements show `522.188 qps` for INT8+FP16 versus `636.729 qps` for FP16, so deployment
retains FP16. See
[`12_yolov8_int8_quantization_engineering/README.md`](12_yolov8_int8_quantization_engineering/README.md).

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

- Ubuntu 22.04 or another recent Linux distribution
- NVIDIA GPU and driver
- CUDA Toolkit
- TensorRT
- C++17 compiler
- CMake 3.10 or newer
- OpenCV
- Python 3 for model export and validation
