# 20 - DeepStream and GStreamer Multi-Stream Analytics

## Purpose

This edge-CV elective maps the lesson 16 architecture to DeepStream: URI sources feed
`nvstreammux`, `nvinfer` executes the TensorRT engine, a custom parser decodes YOLOv8 output, and a
headless sink keeps throughput measurements independent of display rendering.

## Prerequisites

- Use a compatible NVIDIA GPU host and the documented DeepStream 8.0 development container.
- Provide at least two real video sources for runtime acceptance.

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
./20_deepstream_gstreamer_multistream/build_runtime_assets.sh
```

## Run

```bash
python3 20_deepstream_gstreamer_multistream/generate_app_config.py \
  --source /data/camera-a.mp4 \
  --source /data/camera-b.mp4

cd 20_deepstream_gstreamer_multistream/outputs
deepstream-app -c deepstream_app_config.txt
```

The generated `nvstreammux` batch equals the source count. The committed primary-inference config
uses a batch-2 engine, FP16, aspect-ratio preservation, class-aware NMS, and the custom
`NvDsInferParseYoloV8` function. For more than two streams, rebuild the engine/config batch sizes to
match the mux.

## Outputs

- Generated DeepStream configuration, engine, parser library, platform identity, and monitoring
  logs belong under ignored `outputs/`.
- Static validation output alone is not a successful two-stream runtime result.

## Tests

### Verification

Run the static checks before entering the DeepStream container:

```bash
python3 -m unittest discover -s 20_deepstream_gstreamer_multistream/tests -v
python3 20_deepstream_gstreamer_multistream/validate_config.py
```

Acceptance requires the documented DeepStream container, two real videos, parser compilation,
successful model loading, and per-stream FPS output. Save the execution platform beside performance
evidence:

```bash
mkdir -p 20_deepstream_gstreamer_multistream/outputs
nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total \
  --format=csv > 20_deepstream_gstreamer_multistream/outputs/gpu_platform.csv
deepstream-app --version-all \
  > 20_deepstream_gstreamer_multistream/outputs/deepstream_versions.txt
nvidia-smi dmon -s u -d 1 \
  > 20_deepstream_gstreamer_multistream/outputs/gpu_utilization.log
```

Static configuration checks alone are not a successful DeepStream run.

## Checkpoints

1. Configure a DeepStream/GStreamer pipeline for two or more sources and a TensorRT engine.
2. Explain the roles of source, muxer, inference, tracker, OSD, sink, and NVMM memory.
3. Validate generated configuration before running on a compatible DeepStream environment.
