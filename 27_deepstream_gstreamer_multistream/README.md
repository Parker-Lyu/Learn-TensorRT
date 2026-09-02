# 27 - DeepStream and GStreamer Multi-Stream Analytics

## Purpose

This edge-CV elective maps the lesson 19 architecture to DeepStream: URI sources feed
`nvstreammux`, `nvinfer` executes the TensorRT engine, a custom parser decodes YOLOv8 output, and a
headless sink keeps throughput measurements independent of display rendering.

## Prerequisites

- Use a compatible NVIDIA GPU host and the documented DeepStream 8.0 development container.
- The documented DeepStream container includes the two sample videos used by the commands below;
  no additional video download is required.

## Deliverables

- DeepStream application and inference configuration generators
- YOLOv8 DeepStream parser shared library
- Static configuration validation tests and runtime-asset build script

## Pipeline Concepts

```text
uridecodebin x N -> NVMM surfaces -> nvstreammux -> nvinfer -> metadata -> sink
```

`nvstreammux` forms cross-stream batches. `nvinfer` owns TensorRT execution and attaches detection
metadata to each original buffer. Stream identity is carried by DeepStream batch/frame metadata;
the custom parser only converts one network output into boxes and classes.

NVMM can avoid unnecessary host copies between NVIDIA plugins, but decode, colorspace conversion,
muxing, inference, and display choices still determine whether the pipeline remains GPU-resident.

## Setup

DeepStream is not installed in the course's TensorRT development container. Use the NVIDIA
DeepStream 8.0 development image on an x86 NVIDIA host:

```bash
docker run --rm -it --gpus all --network host \
  -v "$PWD:/workspace/Learn-TensorRT" \
  -w /workspace/Learn-TensorRT \
  nvcr.io/nvidia/deepstream:8.0-triton-multiarch
```

TensorRT engines are environment-specific. Build them inside this container with the command in
the `Build` section rather than copying an engine from another TensorRT environment.

## Build

Build the engine and custom parser inside the documented DeepStream container:

```bash
./27_deepstream_gstreamer_multistream/build_runtime_assets.sh
```

The parser build detects the container's CUDA toolkit headers automatically. If the toolkit is
installed in a non-standard location, set `CUDA_HOME` (or `CUDAToolkit_ROOT`) before running the
script, for example `CUDA_HOME=/usr/local/cuda ./27_deepstream_gstreamer_multistream/build_runtime_assets.sh`.

## Run

Generate the two-source DeepStream configuration inside the development container:

```bash
python3 27_deepstream_gstreamer_multistream/generate_app_config.py \
  --source /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 \
  --source /opt/nvidia/deepstream/deepstream/samples/streams/sample_push.mov

# Launch DeepStream with the generated two-stream configuration.
deepstream-app -c 27_deepstream_gstreamer_multistream/outputs/deepstream_app_config.txt
```

Example output (local run):

```text
FileNotFoundError: missing video source: /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4
```

The pinned development container used for this documentation run did not include the official
DeepStream sample media, so `deepstream-app` could not be started. Run the same commands in a
DeepStream container with those sample files mounted to obtain runtime output.

`sample_720p.mp4` and `sample_office.mp4` are real video files shipped in the official DeepStream
container. The two sources exercise the multi-stream mux and batch-2 inference without requiring a
camera or external media. To use your own files, replace both `--source` arguments with paths that
are visible inside the container.

The generated `nvstreammux` batch equals the source count. The committed primary-inference config
uses a batch-2 engine, FP16, aspect-ratio preservation, class-aware NMS, and the custom
`NvDsInferParseYoloV8` function. For more than two streams, rebuild the engine/config batch sizes to
match the mux, using the batch-4 configuration below for the included engine's maximum profile.

### Optional four-stream experiment

The engine built by this lesson has a dynamic batch profile up to four images. To exercise four
concurrent sources with matching `nvstreammux` and `nvinfer` batch sizes, run:

For the optional four-stream batch-4 experiment, generate a matching configuration:

```bash
python3 27_deepstream_gstreamer_multistream/generate_app_config.py \
  --source /opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4 \
  --source /opt/nvidia/deepstream/deepstream/samples/streams/sample_push.mov \
  --source /opt/nvidia/deepstream/deepstream/samples/streams/sample_ride_bike.mov \
  --source /opt/nvidia/deepstream/deepstream/samples/streams/sample_run.mov \
  --inference-config 27_deepstream_gstreamer_multistream/config/config_infer_primary_yolov8_b4.txt

deepstream-app -c 27_deepstream_gstreamer_multistream/outputs/deepstream_app_config.txt
```

This is an optional batch-4 experiment, not an additional acceptance requirement. The current
engine profile supports at most four inputs; larger source counts require rebuilding the engine and
creating a matching inference configuration.

## Outputs

- Generated DeepStream configuration, engine, parser library, platform identity, and monitoring
  logs belong under ignored `outputs/`.
- Static validation output alone is not a successful two-stream runtime result.

## Tests

### Verification

Run the static checks before entering the DeepStream container:

```bash
python3 -m unittest discover -s 27_deepstream_gstreamer_multistream/tests -v
python3 27_deepstream_gstreamer_multistream/validate_config.py
```

Acceptance requires the documented DeepStream container, two decodable video sources, parser
compilation, successful model loading, and per-stream FPS output. Save the execution platform beside performance
evidence:

```bash
mkdir -p 27_deepstream_gstreamer_multistream/outputs
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total \
  --format=csv > 27_deepstream_gstreamer_multistream/outputs/gpu_platform.csv
deepstream-app --version-all \
  > 27_deepstream_gstreamer_multistream/outputs/deepstream_versions.txt
nvidia-smi dmon -s u -d 1 \
  > 27_deepstream_gstreamer_multistream/outputs/gpu_utilization.log
```

Static configuration checks alone are not a successful DeepStream run.

## Checkpoints

1. Configure a DeepStream/GStreamer pipeline for two or more sources and a TensorRT engine.
2. Explain the roles of source, muxer, inference, tracker, OSD, sink, and NVMM memory.
3. Validate generated configuration before running on a compatible DeepStream environment.

## Appendix: DeepStream 基础知识

本附录用最少的概念解释 DeepStream 应用是如何把多路视频送入 GPU 推理流程的。DeepStream 是 NVIDIA 基于 GStreamer 构建的流式分析 SDK，重点不是替代 GStreamer，而是提供面向视频分析的高性能插件、元数据结构和 TensorRT 集成。

### 1. 一条典型的数据流

一个常见的检测管线可以表示为：

```text
source → decode → nvstreammux → nvinfer → nvtracker → nvdsosd → sink
```

- **source / decode**：从文件、RTSP、摄像头或其他 GStreamer 输入产生视频帧，并完成解码。硬件解码器通常输出 GPU 可访问的 NVMM 缓冲区。
- **`nvstreammux`**：接收多路输入，按 batch 组成统一的批次，同时协调不同帧率和到达时间。`batch-size` 通常应与并行源数量及推理配置一致；实时场景还要关注 `live-source`、超时和输入尺寸。
- **`nvinfer`**：调用 TensorRT engine 执行推理。它可以作为 primary GIE 直接处理整帧，也可以作为 secondary GIE 处理上游检测到的目标。
- **`nvtracker`**：根据检测框在连续帧之间关联同一个目标，生成稳定的 `object_id`。跟踪器不会凭空产生检测结果，通常仍需要周期性检测来校正轨迹。
- **`nvdsosd`**：读取元数据并绘制框、标签和置信度。它改变的是显示缓冲区，不是检测结果本身。
- **sink**：将结果显示、编码后保存，或发送到网络。无显示环境可使用文件、编码器或 `fakesink`，不要依赖 GUI 窗口。

### 2. GStreamer 的基本模型

GStreamer 由 **element**（元素）组成，元素通过 **pad** 连接；数据以 buffer 形式沿 pipeline 流动，caps 描述缓冲区的格式、尺寸和帧率。`gst-launch-1.0` 适合快速验证连接关系，生产代码通常使用 C/C++ 或 Python 创建 pipeline、监听 bus 消息并处理 EOS 和错误。

DeepStream 插件名称以 `nv` 开头（例如 `nvv4l2decoder`、`nvstreammux`、`nvinfer`）。元素之间的 caps 不兼容、解码器缺失或动态 pad 未正确连接，都会导致 pipeline 在启动时失败；应优先检查 bus 中的错误信息和插件属性，而不是盲目增加 queue。

### 3. 批处理、时间戳与多路输入

`nvstreammux` 的 batch 是“同一时刻附近收集的多路帧”，不保证所有源严格同步。每个帧和每个目标都带有 `source_id`（或 pad index）以及 presentation timestamp，后处理必须使用这些字段把结果归属回正确的视频源。输入帧率不同或网络抖动时，mux 超时策略会影响延迟和 batch 的完整程度：更长的等待通常提高 batch 利用率，但会增加端到端延迟。

`nvinfer` 的 `batch-size`、engine 的优化 profile 和 mux 的 batch 配置必须互相兼容。动态 batch engine 并不意味着可以无限增加源；仍受 engine 最大 profile、显存、解码吞吐和后续插件能力限制。

### 4. 元数据（Metadata）

DeepStream 使用 `NvDsBatchMeta`、`NvDsFrameMeta` 和 `NvDsObjectMeta` 描述批次、帧和目标。典型关系是：一个 batch 包含多个 frame，一个 frame 属于一个 source，并包含零个或多个 object。插件通过元数据传递检测框、类别、置信度、跟踪 ID 和用户扩展信息，因此应用通常不必复制整帧像素数据。

解析 TensorRT 输出时，parser 负责把网络张量转换为检测框和类别；坐标还需要正确处理缩放、padding（例如保持宽高比时的 letterbox）以及原始帧尺寸。parser 的类别数、输出层名称和配置文件必须与 engine 一致，否则可能出现加载成功但结果错误的情况。

### 5. NVMM、零拷贝与性能

NVMM 是 DeepStream 常用的 NVIDIA 硬件缓冲区内存类型。让解码、缩放、推理和 OSD 尽量在 NVMM/GPU 路径上运行，可以避免频繁的 CPU↔GPU 拷贝。需要在 CPU 访问像素时，才显式映射或转换缓冲区，并及时解除映射。性能分析应同时观察 GPU 利用率、解码负载、batch 等待时间、推理延迟和各路 FPS；单看 GPU 利用率不足以判断 pipeline 是否健康。

### 6. 配置与故障排查顺序

建议按以下顺序定位问题：

1. 用 `gst-inspect-1.0 <plugin>` 确认插件存在及属性名称；
2. 先用单路、短视频验证解码和 caps，再增加 `nvstreammux`；
3. 单独验证 TensorRT engine、parser 和 `nvinfer` 配置的 batch、输入尺寸及标签文件；
4. 再加入 tracker、OSD 和 sink，每次只引入一个变量；
5. 检查 bus 错误、EOS、源文件可读性、GPU/驱动/DeepStream 版本和显存；
6. 对实时源记录端到端延迟和丢帧，而不仅是最终 FPS。

DeepStream 版本、TensorRT 版本、CUDA 驱动和 engine 构建环境需要保持兼容。序列化 engine 通常是特定 GPU、TensorRT 和 profile 的环境产物，不应假定能够跨机器直接复用。

### 7. 与本课的对应关系

本课把多个视频源接入 `nvstreammux`，以 batch 方式交给 primary `nvinfer`，再通过 OSD 和 sink 观察结果。完成实验后，应能解释每个元素的输入输出、batch 中 `source_id` 的作用、NVMM 的性能意义，以及为什么修改源数量时必须同步检查 mux、engine profile 和 inference 配置。
