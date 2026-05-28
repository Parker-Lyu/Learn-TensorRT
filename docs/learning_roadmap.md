# TensorRT Learning Roadmap

This roadmap is designed for an AI algorithm engineer who already understands model training and delivery, but wants to build practical deployment and inference optimization skills for TensorRT, OpenVINO, and C++ inference engineering.

The goal is not to collect demos. The goal is to build a portfolio-quality project that proves you can take a model from PyTorch to a production-like inference pipeline, measure it, optimize it, and explain the trade-offs.

## Target Outcome

After finishing this repository, you should be able to:

- Export a PyTorch YOLO model to ONNX.
- Inspect and simplify ONNX graphs.
- Build TensorRT FP32, FP16, and INT8 engines.
- Write TensorRT C++ inference code without relying on framework wrappers.
- Manage TensorRT, CUDA, and OpenCV resources safely with RAII.
- Implement image preprocessing and detection postprocessing.
- Build a thread-safe producer-consumer inference pipeline.
- Run dynamic batch inference with TensorRT optimization profiles.
- Process multiple video streams with clear scheduling, batching, and dropped-frame policies.
- Debug unsupported operators with ONNX GraphSurgeon and TensorRT plugin strategy.
- Use Nsight Systems to prove where CPU/GPU time is spent.
- Understand DeepStream and GStreamer enough to run multi-stream industrial demos.
- Package C++ inference as a `.so` and call it from Python.
- Measure latency, throughput, memory usage, and accuracy changes.
- Explain performance bottlenecks in CPU preprocessing, GPU inference, memory copies, and synchronization.
- Deploy the same model with OpenVINO for CPU comparison.
- Understand the minimum LLM inference concepts needed for modern deployment interviews.
- Present the project as an interview-ready deployment case study.

## Recommended Pace

Use this roadmap as the default plan. A realistic pace is about 10 weeks if studying full time, or 14 to 16 weeks if studying part time.

| Phase | Time | Focus |
| --- | --- | --- |
| Phase 0 | 2-3 days | Environment and tooling |
| Phase 1 | 1 week | C++, CMake, OpenCV basics |
| Phase 2 | 1-2 weeks | ONNX export and validation |
| Phase 3 | 2 weeks | TensorRT engine build and basic C++ inference |
| Phase 4 | 2 weeks | YOLOv8n end-to-end TensorRT C++ pipeline |
| Phase 5 | 1-2 weeks | Nsight baseline, FP16, INT8, benchmark, accuracy comparison |
| Phase 6 | 2-3 weeks | Producer-consumer pipeline, dynamic batching, async video, multi-stream video |
| Phase 7 | 1 week | CUDA/NPP preprocessing optimization and OpenVINO comparison |
| Phase 8 | 1-2 weeks | Graph surgery, TensorRT plugin strategy, DeepStream, Python binding |
| Phase 9 | 3-5 days | LLM inference entry point and C++ interview katas |
| Phase 10 | 2-3 days | Final report and resume material |

## Directory Plan

### `00_environment_check`

Purpose:

- Record the local machine, driver, CUDA, TensorRT, Python, compiler, and OpenCV versions.
- Keep reproducible commands for checking the environment.

Deliverables:

- `README.md`
- `check_env.sh`
- Optional `env_report.md`

Acceptance criteria:

- You can explain what GPU, driver, CUDA Toolkit, TensorRT, and compiler versions are being used.
- You know whether the environment is host-based or container-based.

### `01_hello_world`

Purpose:

- Build confidence with C++17 and CMake.

Acceptance criteria:

- You can configure, build, and run a small C++ executable.
- You understand `CMakeLists.txt`, target creation, and C++ standard settings.

### `02_opencv_read_show_image`

Purpose:

- Learn image loading and basic OpenCV project setup.

Acceptance criteria:

- You can load an image, check dimensions, and display or save output.
- You can link OpenCV with CMake.

### `03_opencv_preprocess`

Purpose:

- Implement YOLO-style preprocessing outside the model framework.

Topics:

- Resize
- Letterbox
- BGR/RGB conversion
- Normalization
- HWC to CHW
- `float32` host buffer
- Batch buffer layout

Acceptance criteria:

- Given an input image, the program writes a preprocessed tensor or debug image.
- You can explain how image coordinates map back to original image coordinates.

### `04_cuda_memory_stream`

Purpose:

- Learn the CUDA concepts needed for TensorRT inference code.

Topics:

- `cudaMalloc`
- `cudaFree`
- `cudaMemcpy`
- Pinned host memory
- `cudaMallocHost`
- `cudaStream_t`
- Synchronization
- Timing with CUDA events

Acceptance criteria:

- You can copy buffers between host and device.
- You can run a simple async copy/inference-like flow with a stream.
- You can explain why unnecessary synchronization hurts latency.

### `05_torch_to_onnx`

Purpose:

- Export YOLOv8n to ONNX and validate the exported graph.

Topics:

- Ultralytics YOLOv8n export
- ONNX opset
- Static and dynamic shapes
- ONNX Runtime validation
- `onnxsim`
- Netron graph inspection

Acceptance criteria:

- `yolov8n.onnx` is generated.
- ONNX Runtime output is numerically close to PyTorch output for the same image.
- You can identify the model input and output tensor names.

### `06_trtexec_engine`

Purpose:

- Learn TensorRT engine construction before writing C++ code.

Topics:

- `trtexec --onnx`
- FP32 and FP16 engines
- Static shape
- Dynamic shape profiles
- Workspace memory
- Layer profiling
- Engine serialization

Acceptance criteria:

- You can build `.engine` files from ONNX.
- You can run `trtexec` benchmark and read latency, throughput, and memory output.
- You can compare FP32 and FP16 results.

### `07_tensorrt_raii_resource`

Purpose:

- Make TensorRT C++ code exception-safe and long-running-service friendly.

Why it matters:

- Industrial camera systems and edge inference services often run 24x7.
- A small host memory leak, CUDA memory leak, or forgotten TensorRT object can become a production incident.
- RAII proves that resources are released even when early returns or exceptions happen.

Topics:

- RAII
- `std::unique_ptr`
- Custom deleter
- TensorRT object ownership
- CUDA buffer wrapper
- Non-copyable resource classes
- Move semantics for resource handles
- Exception-safe initialization

Example targets:

- `nvinfer1::IRuntime`
- `nvinfer1::ICudaEngine`
- `nvinfer1::IExecutionContext`
- CUDA device buffers
- CUDA streams

Acceptance criteria:

- TensorRT runtime, engine, and context are wrapped by `std::unique_ptr` or small RAII classes.
- CUDA buffers and streams are released automatically.
- The program remains leak-safe if model loading, buffer allocation, or inference setup fails halfway.
- You can explain how RAII protects GPU memory in long-running services.

### `08_tensorrt_cpp_basic`

Purpose:

- Write a minimal TensorRT C++ runtime program.

Topics:

- Logger
- Builder
- ONNX parser
- Runtime creation
- Engine building
- Engine deserialization
- Execution context
- Tensor names and shapes
- Device buffer allocation
- Inference enqueue

Acceptance criteria:

- A C++ program loads a TensorRT engine and runs one inference with dummy or real input.
- Builder, parser, engine, runtime, context, buffer, and stream lifetimes are clear.

### `09_yolov8_trt_python`

Purpose:

- Build a fast debugging reference before the full C++ implementation.

Topics:

- Python TensorRT runtime
- NumPy preprocessing
- Output decoding
- NMS
- Visualization

Acceptance criteria:

- The Python pipeline produces boxes on a test image.
- The output is close to the PyTorch or Ultralytics reference.

### `10_yolov8_trt_cpp`

Purpose:

- Build the main portfolio artifact: end-to-end YOLOv8n TensorRT C++ inference.

Topics:

- OpenCV preprocessing
- TensorRT runtime
- CUDA buffers
- YOLO decode
- NMS
- Coordinate scaling
- Visualization
- CLI arguments

Acceptance criteria:

- The program accepts an image path and engine path.
- It saves an output image with detection boxes.
- It reports preprocessing, inference, postprocessing, and total latency.

### `11_nsight_performance_diagnosis`

Purpose:

- Replace guesswork with timeline-based performance diagnosis.

Why it matters:

- Profiling immediately after the first C++ pipeline gives you a baseline before optimization.
- High-end deployment roles expect evidence: latency tables, profiler traces, and bottleneck explanations.

Topics:

- `trtexec` baseline
- Nsight Systems command-line capture with `nsys`
- Timeline reading
- CPU preprocessing bottleneck
- H2D and D2H copy gaps
- GPU starvation
- CUDA stream overlap verification
- P50/P90/P99 latency reporting

Acceptance criteria:

- You can capture a timeline for the C++ YOLO TensorRT program.
- You can identify whether the GPU is busy or waiting.
- You can explain one optimization using before-and-after timeline evidence.

### `12_yolov8_int8_calibration`

Purpose:

- Learn practical quantization and its trade-offs.

Topics:

- Calibration dataset
- Entropy calibration
- KL divergence intuition
- `IInt8EntropyCalibrator2`
- Calibration table
- INT8 engine build
- Mixed precision fallback
- Sensitive layer fallback to FP16 or FP32
- QAT as a fallback when PTQ fails
- Accuracy comparison
- Latency comparison

Acceptance criteria:

- You can build an INT8 engine.
- You can compare FP32, FP16, and INT8 latency.
- You can explain any visible accuracy drop and propose a fallback strategy.

### `13_cpp_producer_consumer`

Purpose:

- Learn the C++ concurrency pattern behind real camera and video inference systems.

Why it matters:

- A camera may produce frames faster than a model can consume them.
- A single `while` loop hides backpressure, latency buildup, frame dropping policy, and shutdown complexity.

Topics:

- `std::thread`
- `std::mutex`
- `std::condition_variable`
- Thread-safe queue
- Bounded queue
- Producer-consumer pattern
- Backpressure
- Frame dropping strategy
- Graceful shutdown

Acceptance criteria:

- One thread reads images or video frames and pushes them into a bounded thread-safe queue.
- Another thread pops frames and simulates or runs inference.
- The queue has a clear policy when input FPS is higher than inference FPS.
- The program exits cleanly without deadlock.
- You can explain latency versus throughput trade-offs in queue sizing.

### `14_dynamic_batching`

Purpose:

- Learn how to use TensorRT batch dimensions and dynamic optimization profiles for multi-image inference.

Why it matters:

- Medical imaging, offline inspection, and multi-camera systems often benefit from batching.
- TensorRT deployment code must correctly calculate input and output offsets for `N x C x H x W` buffers.

Topics:

- Static batch versus dynamic batch
- TensorRT optimization profiles
- `minShapes`, `optShapes`, and `maxShapes`
- Runtime input shape setting
- Batched preprocessing buffer layout
- Output offset calculation
- Throughput versus latency trade-off

Acceptance criteria:

- A TensorRT engine is built with a dynamic batch profile, for example `1x3x640x640` to `4x3x640x640`.
- C++ code can run batch size 1, 2, and 4 with the same engine.
- Input and output buffer offsets are calculated explicitly.
- A benchmark compares batch size 1 and batch size 4 latency and throughput.

### `15_async_video_pipeline`

Purpose:

- Move from single-image demo to a single-stream production-like inference loop.

Topics:

- Video input
- Frame queue
- Async inference
- Double buffering
- CPU/GPU overlap
- FPS measurement

Acceptance criteria:

- The program can process a video or camera stream.
- You can report average FPS, P50/P90/P99 latency, and GPU utilization.

Additional topics:

- Producer-consumer queue integration
- Dynamic batching from queued frames
- Frame timestamp tracking
- Dropped-frame statistics

### `16_multistream_video_pipeline`

Purpose:

- Handle the real production pattern where one process receives frames from multiple cameras or multiple video files.

Why it matters:

- Industrial inspection, traffic perception, retail analytics, and autonomous driving rigs usually have more than one stream.
- Multi-stream systems need per-stream buffering, global scheduling, batching, overload handling, and per-stream metrics.
- A design that works for one video may fail when four cameras have different FPS, resolution, and jitter.

Reference architecture:

- One capture thread per stream, or a small capture thread pool.
- One bounded queue per stream.
- A scheduler that pulls frames from multiple queues.
- A batch assembler that groups frames into `N x C x H x W`.
- One TensorRT inference worker, or multiple workers if the GPU and model justify it.
- A result dispatcher that sends detections back to the correct stream by `stream_id` and `frame_id`.

Topics:

- Multi-stream input configuration
- Per-stream bounded queues
- Stream ID and frame ID
- Timestamp propagation
- Round-robin scheduling
- Latest-frame-first scheduling
- Micro-batching timeout
- Dynamic batch with partially filled batches
- Per-stream latency and FPS metrics
- Dropped-frame and stale-frame statistics
- Graceful shutdown across multiple threads

Acceptance criteria:

- The program can read from at least two video files or camera-like sources.
- Each stream has independent FPS, queue depth, and dropped-frame counters.
- Frames are batched for TensorRT inference when possible.
- Detection results are routed back to the correct stream.
- The report includes total throughput and per-stream P50/P90/P99 latency.
- You can explain the trade-off between fairness, throughput, and real-time freshness.

### `17_cuda_preprocess_npp`

Purpose:

- Move preprocessing hotspots from CPU OpenCV to GPU-side code when the timeline proves it is useful.

Topics:

- Simple CUDA kernel structure
- BGR to RGB conversion
- Normalization
- HWC to CHW conversion
- Optional resize or letterbox kernel
- NVIDIA NPP overview
- CPU OpenCV versus CUDA preprocessing comparison

Acceptance criteria:

- At least one preprocessing step runs on GPU.
- The result is numerically checked against the OpenCV implementation.
- A benchmark compares CPU preprocessing and GPU/NPP preprocessing.

### `18_openvino_yolov8`

Purpose:

- Compare TensorRT GPU deployment with OpenVINO CPU deployment.

Topics:

- OpenVINO model loading
- CPU inference
- Async infer requests
- FP32/FP16/INT8 where available
- `benchmark_app`

Acceptance criteria:

- The same ONNX model runs with OpenVINO.
- You can compare OpenVINO CPU latency with TensorRT GPU latency.
- You can explain where OpenVINO is relevant for Intel roles.

### `19_onnx_graph_surgery_plugin`

Purpose:

- Learn the escalation path for unsupported operators and graph conversion failures.

Why it matters:

- Real industrial and medical models often contain operators that TensorRT cannot parse directly.
- Senior candidates are expected to know when to change the model, edit the graph, or write a plugin.

Topics:

- Unsupported operator diagnosis
- PyTorch equivalent replacement
- ONNX GraphSurgeon
- Constant folding
- Node replacement and node splitting
- TensorRT plugin design
- `IPluginV2DynamicExt` responsibilities
- Plugin serialization and deserialization

Acceptance criteria:

- You can explain the three-level strategy: model rewrite, ONNX graph surgery, TensorRT plugin.
- You can edit a small ONNX graph with GraphSurgeon.
- You can describe the key plugin lifecycle methods: `getOutputDimensions`, `configurePlugin`, `enqueue`, `serialize`, and `clone`.

### `20_deepstream_gstreamer_multistream`

Purpose:

- Learn the industrial multi-stream stack used around TensorRT in NVIDIA edge deployments.

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

- A DeepStream sample runs successfully on the local machine or a compatible environment.
- A TensorRT engine is used through DeepStream configuration.
- At least two video streams are processed concurrently.
- You can explain where TensorRT sits inside the GStreamer pipeline.

### `21_cpp_shared_library_python_binding`

Purpose:

- Package C++ inference code so higher-level Python business logic can call it.

Why it matters:

- Production systems often keep the fast inference core in C++ and orchestration/business state in Python.

Topics:

- C ABI wrapper
- `.so` dynamic library
- Header design
- Struct-based input/output
- `ctypes`
- `pybind11`
- Ownership across language boundaries
- Error code versus exception boundary

Acceptance criteria:

- The TensorRT C++ inference class is compiled into a shared library.
- A Python script loads the library and runs inference.
- The exposed API uses simple inputs and structured outputs.

### `22_llm_inference_intro`

Purpose:

- Build an entry-level understanding of LLM inference so deployment interviews do not stop at CNN/CV models.

Scope:

- This is not the main project line.
- The goal is to understand core concepts and run a small local example, not to build a full LLM serving stack.

Topics:

- Tokenization
- Prefill and decode
- KV cache
- Batch size versus sequence length
- Throughput versus first-token latency
- FP16, INT8, INT4, and weight-only quantization
- ONNX Runtime, OpenVINO GenAI, TensorRT-LLM, vLLM, and llama.cpp at a high level
- CPU/GPU memory limits

Acceptance criteria:

- You can run a small local LLM or read through one minimal inference example.
- You can explain why LLM inference bottlenecks differ from YOLO inference.
- You can explain KV cache and why decode is often memory-bandwidth bound.
- You know when to mention TensorRT-LLM, OpenVINO GenAI, vLLM, or llama.cpp in interviews.

### `23_cpp_interview_katas`

Purpose:

- Prepare for practical C++ interview questions related to CV deployment instead of generic puzzle-style questions.

Topics:

- IoU
- NMS
- Bilinear interpolation
- Letterbox coordinate mapping
- HWC to CHW memory reorder
- Thread-safe queue
- Top-K filtering
- Simple ring buffer
- RAII wrapper for CUDA memory

Acceptance criteria:

- Each kata has a small C++ implementation and test input.
- You can write IoU, NMS, letterbox mapping, and a bounded queue without looking up code.

### `24_benchmark_report`

Purpose:

- Convert the learning project into interview material.

Deliverables:

- `report.md`
- Latency table
- Throughput table
- Accuracy notes
- Environment table
- Bottleneck analysis
- Future work

Acceptance criteria:

- A recruiter or interviewer can understand the project in five minutes.
- You can defend every number in the report.

## Suggested Weekly Routine

For each lesson:

1. Read the API or tool documentation.
2. Implement the smallest runnable version.
3. Add command examples to the lesson README.
4. Add timing or correctness checks.
5. Write three notes: what worked, what failed, what changed your mental model.

## Interview-Oriented Checkpoints

After `06_trtexec_engine`, you should be able to answer:

- What is a TensorRT engine?
- Why is an engine hardware-specific?
- What changes when enabling FP16?
- What is a dynamic shape optimization profile?
- What does `trtexec` measure?

After `07_tensorrt_raii_resource`, you should be able to answer:

- What is RAII?
- Why are raw TensorRT pointers risky in production code?
- How do you use `std::unique_ptr` with a custom deleter?
- How do you make CUDA buffers exception-safe?
- What happens if engine loading fails halfway through initialization?

After `10_yolov8_trt_cpp`, you should be able to answer:

- How does YOLO preprocessing work?
- How do you allocate and bind TensorRT input/output buffers?
- Where are the main latency costs?
- What parts run on CPU and what parts run on GPU?
- How do you validate TensorRT output against PyTorch output?

After `13_cpp_producer_consumer`, `14_dynamic_batching`, and `16_multistream_video_pipeline`, you should be able to answer:

- Why is a single video-reading loop not enough for industrial camera systems?
- How do `std::mutex` and `std::condition_variable` work together?
- What should happen when the producer is faster than the consumer?
- How do you build a TensorRT engine with dynamic batch profiles?
- How do you calculate offsets for batched `NCHW` input buffers?
- How do you keep stream identity when frames from multiple cameras are batched together?
- How do you choose between fairness and latest-frame freshness?
- What metrics do you report for each stream?

After `22_llm_inference_intro`, you should be able to answer:

- What are prefill and decode?
- What is KV cache?
- Why is LLM decoding often memory-bandwidth bound?
- How is LLM batching different from YOLO image batching?
- What problems do TensorRT-LLM, OpenVINO GenAI, vLLM, and llama.cpp try to solve?

After `19_onnx_graph_surgery_plugin`, you should be able to answer:

- What do you do when TensorRT does not support an ONNX operator?
- When should you rewrite PyTorch code instead of writing a plugin?
- What does ONNX GraphSurgeon do?
- What are the core responsibilities of a TensorRT dynamic plugin?

After `11_nsight_performance_diagnosis` and `17_cuda_preprocess_npp`, you should be able to answer:

- How do you prove GPU starvation from a timeline?
- How do pinned memory and async copies affect overlap?
- When is GPU preprocessing worth the added complexity?
- How do you compare two optimization attempts fairly?

After `20_deepstream_gstreamer_multistream` and `21_cpp_shared_library_python_binding`, you should be able to answer:

- What is a GStreamer pipeline?
- What does `nvstreammux` do?
- What does zero-copy mean in DeepStream?
- How do you expose a C++ TensorRT engine to Python safely?

After `24_benchmark_report`, you should be able to answer:

- What is the best latency you achieved on this RTX 2060?
- How much faster is FP16 than FP32?
- Did INT8 help on this model and GPU?
- What accuracy trade-off did you observe?
- What would you change for production deployment?

## Portfolio Story

The final story should be:

I took YOLOv8n from PyTorch to ONNX, validated it, built TensorRT FP32/FP16/INT8 engines, implemented a C++ inference pipeline with RAII-managed TensorRT/CUDA resources, OpenCV preprocessing, YOLO postprocessing, producer-consumer video flow, dynamic batching, and multi-stream video scheduling, measured latency and throughput on RTX 2060, diagnosed bottlenecks with Nsight Systems, explored graph surgery and plugin strategy for unsupported operators, compared the same model with OpenVINO CPU inference, ran a DeepStream-style multi-stream path, exposed C++ inference to Python, learned the entry-level LLM inference concepts, and summarized the engineering trade-offs in a benchmark report.
