# 20 - DeepStream GStreamer Multi-Stream

Goal: understand the NVIDIA production-style video analytics stack around TensorRT.

Topics:

- DeepStream application structure
- GStreamer pipeline concepts
- Source, muxer, infer, tracker, OSD, sink
- `nvstreammux`
- `nvinfer`
- TensorRT engine integration
- Zero-copy and NVMM memory concepts
- Multi-stream configuration
- GPU memory and FPS monitoring

Acceptance criteria:

- A DeepStream sample runs on the local or target environment.
- A TensorRT engine is used through DeepStream configuration.
- At least two streams are processed concurrently.
- You can explain where TensorRT sits inside the GStreamer pipeline.
