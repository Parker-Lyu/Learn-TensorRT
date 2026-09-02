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

## Appendix: DeepStream Fundamentals

This appendix introduces the minimum concepts needed to understand how DeepStream moves
multiple video streams through a GPU inference pipeline. DeepStream is NVIDIA's high-performance
streaming analytics SDK built on GStreamer. It does not replace GStreamer; instead, it provides
video-analytics plugins, metadata structures, and TensorRT integration.

### 1. A Typical Data Flow

A common detection pipeline looks like this:

```text
source → decode → nvstreammux → nvinfer → nvtracker → nvdsosd → sink
```

- **source / decode**: Reads video from files, RTSP, cameras, or other GStreamer inputs and decodes
  frames. Hardware decoders usually produce GPU-accessible NVMM buffers.
- **`nvstreammux`**: Receives multiple inputs and assembles batches while coordinating different
  frame rates and arrival times. `batch-size` should normally match the number of parallel sources
  and the inference configuration. For live streams, also consider `live-source`, timeout, and
  input dimensions.
- **`nvinfer`**: Runs TensorRT inference. As a primary GIE it processes full frames; as a secondary
  GIE it can process objects detected upstream.
- **`nvtracker`**: Associates detections across frames and assigns stable `object_id` values. A
  tracker does not create detections by itself; periodic detection is still commonly required to
  correct tracks.
- **`nvdsosd`**: Reads metadata and draws boxes, labels, and confidence values. It changes the
  display buffer, not the detection metadata itself.
- **sink**: Displays results, writes encoded video, or sends data over a network. In headless
  environments, use a file sink, encoder, or `fakesink` rather than relying on a GUI window.

### 2. The GStreamer Model

GStreamer pipelines are built from **elements** connected through **pads**. Data travels as buffers,
while caps describe buffer format, dimensions, and frame rate. `gst-launch-1.0` is useful for quickly
checking connections; production applications generally create the pipeline in C/C++ or Python,
monitor bus messages, and handle EOS and errors.

DeepStream plugins commonly begin with `nv` (for example, `nvv4l2decoder`, `nvstreammux`, and
`nvinfer`). Incompatible caps, missing decoders, or incorrectly linked dynamic pads can make a
pipeline fail at startup. Check bus errors and plugin properties before adding queues blindly.

### 3. Batching, Timestamps, and Multiple Inputs

An `nvstreammux` batch contains frames collected from several sources around the same time; it does
not guarantee strict synchronization. Every frame and object carries a source identifier (source ID
or pad index) and a presentation timestamp. Postprocessing must use these fields to associate each
result with the correct video source. With different frame rates or network jitter, mux timeout
settings affect both batch completeness and latency: waiting longer may improve batch utilization,
but increases end-to-end delay.

The `batch-size` values of `nvinfer`, the engine optimization profile, and the mux configuration must
be compatible. A dynamic-batch engine cannot support unlimited sources: the maximum profile, GPU
memory, decoder throughput, and downstream plugin capacity still impose limits.

### 4. Metadata

DeepStream uses `NvDsBatchMeta`, `NvDsFrameMeta`, and `NvDsObjectMeta` to describe batches, frames,
and objects. A batch contains multiple frames; each frame belongs to one source and can contain zero
or more objects. Plugins pass detection boxes, classes, confidences, tracking IDs, and user-defined
fields through metadata, so applications usually do not need to copy full image buffers.

The TensorRT output parser converts network tensors into detection boxes and classes. Coordinates
must account for scaling, padding (for example, letterboxing used to preserve aspect ratio), and the
original frame dimensions. Parser class counts, output-layer names, and configuration values must
match the engine; otherwise an engine may load successfully while producing incorrect results.

### 5. NVMM, Zero-Copy, and Performance

NVMM is the NVIDIA hardware-buffer memory type commonly used by DeepStream. Keeping decode,
scaling, inference, and OSD on the NVMM/GPU path minimizes CPU-to-GPU copies. When CPU pixel access
is necessary, explicitly map or convert the buffer and unmap it promptly. Performance analysis
should consider GPU utilization, decoder load, batch wait time, inference latency, and per-stream
FPS. GPU utilization alone does not indicate whether a pipeline is healthy.

### 6. Configuration and Troubleshooting Order

Use this order when diagnosing a pipeline:

1. Run `gst-inspect-1.0 <plugin>` to confirm that the plugin exists and verify property names.
2. Validate decoding and caps with one source and a short video before adding `nvstreammux`.
3. Verify the TensorRT engine, parser, and `nvinfer` settings, including batch size, input dimensions,
   and label files.
4. Add tracker, OSD, and sink one at a time so each change is isolated.
5. Check bus errors, EOS, source readability, GPU/driver/DeepStream versions, and available memory.
6. For live sources, record end-to-end latency and dropped frames, not only final FPS.

DeepStream, TensorRT, CUDA, and driver versions must remain compatible. Serialized engines are usually
artifacts of a particular GPU, TensorRT version, and optimization profile; do not assume that an
engine can be reused on another machine without rebuilding it.

### 7. Relation to This Lesson

This lesson connects multiple video sources to `nvstreammux`, sends batches to a primary `nvinfer`,
and observes results through OSD and a sink. After completing the experiment, you should be able to
explain each element's inputs and outputs, the role of `source_id` in a batch, the performance value
of NVMM, and why changing the source count requires checking the mux, engine profile, and inference
configuration together.
