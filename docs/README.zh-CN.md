[English](../README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

# Learn TensorRT：面向生产的 C++ 推理与优化

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.14.1-green.svg)](https://developer.nvidia.com/tensorrt)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Container](https://img.shields.io/badge/NGC_PyTorch-25.11-orange.svg)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

这是一门注重实践的 C++17 课程，聚焦于 **TensorRT 10.14 高性能部署**和
**低延迟流水线工程**，涵盖 **YOLOv8、CUDA 预处理、INT8 量化、异步流水线、Triton、DeepStream 和 Nsight 性能分析**。

使用 C++17 从导出的模型开始，构建一条完整且可度量的 YOLOv8 部署流水线。你将学习使系统可靠的工程实践：
基于 RAII 的所有权管理、明确的资源生命周期、基于目标的 CMake、防御性验证、可复现的证据，以及能够超越单一可执行程序持续演进的组件。

沿着性能关键路径完成端到端实践：检查并修复 ONNX 图，使用动态形状构建 TensorRT 引擎，
通过 CUDA 和 NPP 高效传输数据，并实现 YOLOv8 后处理。随后通过显式 Q/DQ INT8 量化、
异步多流调度、Sanitizer 与浸泡测试以及 Nsight 性能分析来优化和运行系统。进阶选修内容包括插件、
Triton、DeepStream、Jetson DLA、OpenVINO、Python 绑定和 LLM 推理。最终你将获得可运行、
可度量的部署成果，而不只是一些 `trtexec` 命令。

## 🎬 项目概览

[![观看 Learn TensorRT 项目概览](../assets/project-overview-video-thumbnail.jpg)](https://www.youtube.com/watch?v=K1spF7xL_0o)

在 [YouTube](https://www.youtube.com/watch?v=K1spF7xL_0o) 上观看项目简介，了解课程范围、
部署工作流，以及从 PyTorch 走向生产级 TensorRT 推理的工程路径。


## 目标读者

本仓库面向希望跨越算法研究与生产工程之间鸿沟的开发者：

* **正在转向部署的算法工程师：** 你已具备扎实的深度学习知识（PyTorch、ONNX），但需要掌握面向生产级推理的 C++/TensorRT。
* **初级/中级部署工程师：** 你使用过 Python TensorRT API 或 `trtexec`，并希望进一步掌握显式 Q/DQ 量化、CUDA 零拷贝、多流异步执行和 Nsight 性能分析。
* **构建生产级作品集的工程师：** 你的目标是高级边缘计算机视觉或高性能 AI 部署岗位，需要一个能够展示真实流水线架构的参考项目。

**先修要求：** 这是一门内容较为深入的 C++ 工程课程。你应当基本熟悉 ISO C++17 和 CMake。
只使用过 Python 的开发者应先学习 C++ 所有权、并发、CUDA 执行和构建系统基础。


## 学习路线

你可以先阅读 [`docs/learning_roadmap.md`](learning_roadmap.md)。使用
[`docs/coverage_matrix.md`](coverage_matrix.md) 将核心与选修主题对应到生产部署技能。
你也可以直接按顺序完成课程。


项目由一系列小型课程组成。每节课都应产出一个可运行的成果和一份简短的学习记录。
共享图片及其他可复用资源位于根目录的 `assets` 文件夹中。

## 使用方法
本课程在 Ubuntu 上开发和测试。Windows 或 WSL 环境可能可以运行，但它们不是参考环境，可能需要额外排查问题。

1. 在具备可用 NVIDIA GPU 的 Ubuntu 机器上克隆本仓库：

```bash
git clone git@github.com:Parker-Lyu/Learn-TensorRT.git
cd Learn-TensorRT
```

2. 按照 [`00_environment_check/README.md`](../00_environment_check/README.md) 启动固定版本的开发环境。

3. 你也可以安装 VS Code 和 Dev Containers 扩展，连接到正在运行的容器，然后打开已挂载的项目目录。

4. 按顺序完成核心课程，并在每个检查点撰写报告。之后根据目标岗位的职位描述选择选修内容，
而不是按编号依次完成所有选修课。

### 核心路线

| 阶段 | 课程 | 成果 |
| --- | --- | --- |
| 基础 | `00_environment_check` 至 `11_yolov8_trt_cpp` | 端到端 YOLOv8n TensorRT C++ 推理 |
| 检查点 1 | `12_end_to_end_validation_report` | 可复现的功能与架构证据 |
| 优化 | `13_nsight_performance_diagnosis`、`14_yolov8_int8_quantization_engineering` | 经过性能分析的 FP32/FP16/INT8 表现、显式 Q/DQ 量化和部署证据 |
| 检查点 2 | `15_precision_performance_report` | 可用于求职申请的精度与性能报告 |
| 流水线 | `16_cpp_producer_consumer` 至 `21_integrated_tensorrt_video_pipeline` | 与 CUDA/NPP 预处理及真实 TensorRT 执行集成的有界异步/多流编排 |
| 检查点 3 | `22_pipeline_performance_report` | 可复现的负载、延迟、吞吐量、Sanitizer、重启和 30 分钟浸泡测试报告；状态按环境生成 |


### 选修方向

课程 `23`–`31` 使用全局稳定编号。编号并不意味着每门选修课都应按数字顺序完成。
请遵循下面各方向的顺序和先修要求；课程 `32` 是最终综合项目。

| 方向 | 课程 | 推荐顺序与先修要求 |
| --- | --- | --- |
| 服务端推理 | `24_triton_inference_server`、`29_cpp_shared_library_python_binding` | 两者都基于课程 17，彼此独立。 |
| 边缘计算机视觉 | `27_deepstream_gstreamer_multistream`、`28_jetson_orin_xavier_dla_deployment` | 条件允许时先学习 27 再学习 28；完整验证 28 需要 Jetson/DLA 硬件。 |
| 高级 TensorRT | `25_onnx_graph_surgery_plugin`、`26_custom_tensorrt_plugin` | **25 → 26**：先使用 GraphSurgeon，再使用 TensorRT 插件修复同一个不受支持的 `com.acme::AcmeSwish` 模型。 |
| CPU 与 Intel | `23_openvino_yolov8` | 独立分支；按文档复用课程 05/15 的 ONNX 和对比证据。 |
| LLM 基础认知 | `30_llm_inference_intro` | 独立分支；不要求先完成 YOLO/TensorRT 选修课。 |
| CUDA 内核优化 | `31_nsight_compute_kernel_analysis` | 完成课程 13，并理解课程 26 的自有内核边界；分析一个小型 MLP、优化 LayerNorm，并做出算子/网络/生产决策。 |
| 最终综合 | `32_final_portfolio_case_study` | 完成核心检查点以及你计划展示的选修证据后再进行。 |


## 学习方法

按照每节课的 `README.md` 构建并运行代码，然后检查输出。

获得预期输出只是开始。将 AI 作为学习辅助工具时，应以批判性的态度研究每节课的代码，
努力理解其实现，而不只是让程序成功运行。

## 环境要求

- 一台配备合适 NVIDIA GPU、能够运行 Docker 和 NVIDIA 容器的 Ubuntu 主机。


## 🚀 从学习到入职

[![职业机会](https://img.shields.io/badge/Career-Land_Your_Dream_Role-brightgreen.svg?style=for-the-badge&logo=rocket)](https://github.com/Parker-Lyu/Learn-TensorRT)

祝你编码顺利，也希望本仓库能帮助你获得理想的高性能 AI 工程职位！🎉

## 关于作者

Parker Lyu — [LinkedIn](https://www.linkedin.com/in/parker-lyu-ai/)

## 许可证

本项目采用 [MIT License](../LICENSE)。
