# TensorRT Learning Roadmap

This roadmap is designed for an AI algorithm engineer who already understands model training and delivery, but wants to build practical deployment and inference optimization skills for TensorRT, OpenVINO, and C++ inference engineering.

The goal is not to collect demos. The goal is to build a portfolio-quality project that proves you can take a model from PyTorch to a production-like inference pipeline, measure it, optimize it, and explain the trade-offs.

## Target Outcome

After finishing this repository, you should be able to:

- Export a PyTorch YOLO model to ONNX.
- Inspect and simplify ONNX graphs.
- Build TensorRT FP32, FP16, and INT8 engines.
- Use Polygraphy to compare ONNX Runtime and TensorRT outputs for controlled single-input
  precision alignment, then extend validation to multi-image accuracy regression.
- Write TensorRT C++ inference code without relying on framework wrappers.
- Manage TensorRT, CUDA, and OpenCV resources safely with RAII.
- Implement image preprocessing and detection postprocessing.
- Build a thread-safe producer-consumer inference pipeline.
- Run dynamic batch inference with TensorRT optimization profiles.
- Process multiple video streams with clear scheduling, batching, and dropped-frame policies.
- Debug unsupported operators with ONNX GraphSurgeon and a runnable TensorRT custom plugin.
- Use Nsight Systems to prove where CPU/GPU time is spent.
- Understand DeepStream and GStreamer enough to run multi-stream industrial demos.
- Understand Jetson Orin/Xavier deployment, cross compilation, and DLA constraints.
- Package C++ inference as a `.so` and call it from Python.
- Measure latency, throughput, memory usage, tensor drift, and task-level accuracy changes.
- Explain performance bottlenecks in CPU preprocessing, GPU inference, memory copies, and synchronization.
- Deploy the same model with OpenVINO for CPU comparison.
- Understand the minimum LLM inference concepts needed for modern deployment interviews.
- Present the project as an interview-ready deployment case study.

## Engineering Architecture Track

The early lessons stay small so each topic is runnable and understandable. As the course progresses,
the same code should grow toward a portfolio-quality C++ deployment project instead of remaining a
set of disconnected demos.

Architectural habits to carry through the course:

- Keep reusable logic behind small C++ APIs, with headers and source files separated when the lesson
  has more than one concept.
- Prefer CMake library targets for reusable components, then link a small executable for the lesson
  artifact.
- Add focused tests for reusable algorithms and resource wrappers once a lesson has meaningful edge
  cases.
- Use the pinned TensorRT Dev Container as the normal development environment; introduce Dockerfile
  authoring later when packaging and runtime delivery become the lesson goal.
- Treat Unified Memory and zero-copy-style paths as performance trade-offs to measure, not as magic
  shortcuts. `cudaMallocManaged` simplifies ownership but can still page migrate on discrete GPUs.

By the final portfolio stage, the strongest version of the project should resemble this shape:

```text
industrial_deployment_trtexec/
├── .github/workflows/          # CI for CMake configure/build/tests where runners support it.
├── cmake/                      # Reusable CMake helpers for OpenCV, TensorRT, CUDA, and warnings.
├── src/
│   ├── preprocess/             # Letterbox, layout conversion, CPU/CUDA/NPP preprocessing variants.
│   ├── inference/              # TensorRT runtime, engine, context, buffers, and stream wrappers.
│   ├── postprocess/            # Decode, NMS, coordinate mapping, and result formatting.
│   ├── pipeline/               # Producer-consumer, batching, async, and multi-stream scheduling.
│   └── app/                    # Small executables that assemble the reusable components.
├── tests/                      # Google Test cases for reusable algorithms and resource wrappers.
├── tools/                      # Export, engine-build, benchmark, and report helpers.
├── Dockerfile                  # Lean runtime image introduced near the final packaging lesson.
└── CMakeLists.txt
```

This final structure is a destination, not a reason to overload lesson 01. Each lesson should still
produce one runnable artifact and one concise README.

## Learning Flow

Use this roadmap as the default plan. The modules are ordered by dependency and interview story, not by calendar time.

| Phase | Focus |
| --- | --- |
| Phase 0 | Environment and tooling |
| Phase 1 | C++, CMake, OpenCV basics |
| Phase 2 | ONNX export, validation, and single-input precision alignment |
| Phase 3 | TensorRT engine build and basic C++ inference |
| Phase 4 | YOLOv8n end-to-end TensorRT C++ pipeline |
| Phase 5 | Nsight baseline, FP16, INT8, benchmark, and multi-image accuracy comparison |
| Phase 6 | Producer-consumer pipeline, dynamic batching, async video, and multi-stream video |
| Phase 7 | CUDA/NPP preprocessing optimization and OpenVINO comparison |
| Phase 8 | Graph surgery, runnable TensorRT plugin, DeepStream, Jetson/DLA, and Python binding |
| Phase 9 | LLM inference entry point and C++ interview katas |
| Phase 10 | Final report and resume material |

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
- Start separating reusable preprocessing logic from the executable entry point.

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
- Letterbox and coordinate-mapping helpers validate invalid inputs and are easy to test from a
  focused test target.

### `04_cuda_memory_stream`

Purpose:

- Learn the CUDA concepts needed for TensorRT inference code.

Topics:

- `cudaMalloc`
- `cudaFree`
- `cudaMemcpy`
- Pinned host memory
- `cudaMallocHost`
- Mapped pinned memory with `cudaHostAllocMapped`
- Unified Memory with `cudaMallocManaged`
- Explicit copy, mapped access, and managed memory trade-offs
- `cudaStream_t`
- Synchronization
- Timing with CUDA events

Acceptance criteria:

- You can copy buffers between host and device.
- You can run a simple async copy/inference-like flow with a stream.
- You can explain why unnecessary synchronization hurts latency.
- You can explain why mapped pinned memory can remove explicit `cudaMemcpy` calls but still uses PCIe bandwidth on discrete GPUs.
- You can explain why Unified Memory is convenient but not automatically low-latency.

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

### `06a_polygraphy_precision_alignment`

Purpose:

- Learn a repeatable single-input precision-debug workflow when ONNX Runtime and TensorRT outputs
  disagree.

Why it matters:

- Real deployment work is not finished when an engine builds successfully.
- Senior candidates should be able to prove where numerical drift starts instead of guessing whether preprocessing, export, precision mode, or TensorRT parsing caused the issue.
- A one-image tensor comparison is a debugging gate, not a dataset-level release criterion. Later
  lessons extend it into multi-image drift statistics and decoded detection-quality comparison.

Topics:

- Polygraphy model inspection
- ONNX Runtime versus TensorRT comparison
- Saving and comparing one controlled input tensor and its raw model output
- Layerwise or tensorwise debug workflow
- FP32 and FP16 drift analysis for a controlled sample
- Tolerance selection for deployment reports
- Reproducible command logs for interview discussion

Acceptance criteria:

- You can run Polygraphy against the YOLO ONNX model and a TensorRT engine.
- You can compare ONNX Runtime and TensorRT outputs for the same single input tensor.
- You can save mismatch evidence and explain whether the drift comes from export, preprocessing, precision mode, or TensorRT conversion.
- You can write a short precision-alignment note that belongs in the final benchmark report and
  clearly states that multi-image detection validation still follows.

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
- Begin converging lesson code into reusable preprocessing, inference, and postprocessing modules.

Topics:

- OpenCV preprocessing
- TensorRT runtime
- CUDA buffers
- YOLO decode
- NMS
- Coordinate scaling
- Visualization
- CLI arguments
- Library targets for reusable components
- Focused tests for preprocessing and postprocessing edge cases

Acceptance criteria:

- The program accepts an image path and engine path.
- It saves an output image with detection boxes.
- It reports preprocessing, inference, postprocessing, and total latency.
- Reusable preprocessing, inference, and postprocessing code is not trapped inside `main`.
- Focused tests cover representative invalid input and boundary cases.

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

- Learn practical quantization and its trade-offs using both speed and accuracy evidence.

Topics:

- Calibration dataset
- Validation image set separate from calibration data
- Entropy calibration
- KL divergence intuition
- `IInt8EntropyCalibrator2`
- Calibration table
- INT8 engine build
- FP32, FP16, and INT8 tensor drift summary across multiple images
- Decoded box, class, and confidence comparison
- Mixed precision fallback
- Sensitive layer fallback to FP16 or FP32
- QAT as a fallback when PTQ fails
- Accuracy comparison
- Latency comparison

Acceptance criteria:

- You can build an INT8 engine.
- You can compare FP32, FP16, and INT8 latency.
- You can compare FP32, FP16, and INT8 detection quality on a small representative validation set.
- You can list changed detections or high-drift examples that deserve visual inspection.
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
- Host-device transfer strategies
- Mapped pinned memory trade-offs
- Unified Memory prefetching and page migration risk
- Decode/capture-to-GPU paths such as NVDEC, DeepStream NVMM, or GPUDirect where available
- CPU OpenCV versus CUDA preprocessing comparison

Acceptance criteria:

- At least one preprocessing step runs on GPU.
- The result is numerically checked against the OpenCV implementation.
- A benchmark compares CPU preprocessing and GPU/NPP preprocessing.
- Transfer time is measured separately from preprocessing and inference time.
- You can explain whether a zero-copy-style path is actually faster on the tested hardware.

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

### `19a_custom_tensorrt_plugin`

Purpose:

- Build one runnable custom TensorRT plugin instead of only describing the plugin strategy.

Why it matters:

- Custom plugin experience is a strong signal for roles that deploy non-standard CV, medical, industrial, or research models.
- A small complete plugin demonstrates C++ ABI awareness, CUDA kernel integration, TensorRT lifecycle knowledge, and numerical validation.

Suggested plugin scope:

- Implement a compact operator such as `ScaleShift`, `Clip`, or `CustomNormalize`.
- Keep the operator simple enough that plugin mechanics, serialization, and validation remain the teaching focus.

Topics:

- `IPluginV2DynamicExt`
- Plugin creator registration
- CUDA kernel launch from `enqueue`
- Dynamic shape and data type handling
- Plugin field collection and parameters
- Serialization and deserialization
- Building a plugin shared library
- Loading plugins with `trtexec --plugins`
- Loading plugins from C++ runtime code
- ONNX GraphSurgeon replacement with a plugin node
- Polygraphy or ONNX Runtime reference comparison

Acceptance criteria:

- A plugin shared library builds with CMake.
- `trtexec` can load the plugin library and build an engine containing the plugin layer.
- A small C++ or Python runtime example loads the plugin-backed engine and runs inference.
- The plugin output is numerically checked against a CPU/Python reference.
- You can explain the plugin lifecycle from registration to `enqueue` to engine deserialization.

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

### `20a_jetson_orin_xavier_dla_deployment`

Purpose:

- Understand how TensorRT deployment changes on Jetson Orin/Xavier edge devices, especially when DLA is involved.

Scope:

- This is an edge-deployment extension. It can be documented on x86 first and fully verified later on a Jetson target.

Why it matters:

- Many CV deployment roles involve edge boxes, robotics, industrial cameras, or embedded NVIDIA platforms rather than only desktop GPUs.
- Jetson work requires version discipline because JetPack, CUDA, TensorRT, cuDNN, DeepStream, and kernel drivers are tightly coupled.

Topics:

- JetPack, CUDA, TensorRT, cuDNN, and DeepStream version compatibility
- Native Jetson build versus x86-to-aarch64 cross compilation
- CMake toolchain file for aarch64 targets
- Container versus bare-metal deployment on Jetson
- DLA-supported layer constraints
- `trtexec --useDLACore`
- GPU fallback behavior
- FP16 and INT8 on DLA
- Power modes, clocks, thermals, and memory bandwidth
- Orin/Xavier benchmark notes and deployment checklist

Acceptance criteria:

- The lesson documents the target Jetson hardware, JetPack version, TensorRT version, power mode, and clocks.
- The project has a clear native-build path and a cross-compilation checklist for aarch64.
- A YOLO TensorRT engine is attempted with DLA, with unsupported layers and GPU fallback recorded.
- Latency, throughput, memory, and power-mode notes are compared against the desktop GPU baseline when hardware is available.
- If no Jetson target is available, the lesson still records the exact commands and expected validation steps for future hardware verification.

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

- Each kata has a small C++ implementation and either a focused executable check or Google Test
  coverage.
- You can write IoU, NMS, letterbox mapping, and a bounded queue without looking up code.
- Destructive edge cases are covered for empty inputs, extreme coordinates, overlapping boxes, and
  queue boundary behavior.

### `24_benchmark_report`

Purpose:

- Convert the learning project into interview material.

Deliverables:

- `report.md`
- Latency table
- Throughput table
- Accuracy notes that separate single-input tensor alignment, multi-image drift statistics, and
  decoded detection-quality comparison
- Environment table
- Test evidence table
- CI/build notes
- Production Dockerfile
- Development image versus runtime image size comparison
- Bottleneck analysis
- Future work

Acceptance criteria:

- A recruiter or interviewer can understand the project in five minutes.
- You can defend every number in the report.
- You can explain why a single-input Polygraphy pass is useful but not sufficient for release
  approval.
- The final report points to the reusable module structure and the tests that protect core
  preprocessing, postprocessing, and resource-management behavior.
- CI or a documented local equivalent configures, builds, and runs the available tests.
- A multi-stage Docker build produces a runtime image that contains only the files needed to run inference.

## Suggested Lesson Routine

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

After `06a_polygraphy_precision_alignment`, you should be able to answer:

- Why can ONNX Runtime and TensorRT produce different outputs?
- How do you compare two backends with the same input tensor?
- How do you decide whether a mismatch is acceptable numerical drift or a deployment bug?
- Why is one-image tensor alignment insufficient for release approval?
- How would you debug the first layer where accuracy starts to diverge?

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

After `19_onnx_graph_surgery_plugin` and `19a_custom_tensorrt_plugin`, you should be able to answer:

- What do you do when TensorRT does not support an ONNX operator?
- When should you rewrite PyTorch code instead of writing a plugin?
- What does ONNX GraphSurgeon do?
- What are the core responsibilities of a TensorRT dynamic plugin?
- How is a plugin registered, serialized, deserialized, and called from `enqueue`?
- How do you validate plugin output against a reference implementation?

After `11_nsight_performance_diagnosis` and `17_cuda_preprocess_npp`, you should be able to answer:

- How do you prove GPU starvation from a timeline?
- How do pinned memory and async copies affect overlap?
- When is GPU preprocessing worth the added complexity?
- How do you compare two optimization attempts fairly?

After `20_deepstream_gstreamer_multistream`, `20a_jetson_orin_xavier_dla_deployment`, and `21_cpp_shared_library_python_binding`, you should be able to answer:

- What is a GStreamer pipeline?
- What does `nvstreammux` do?
- What does zero-copy mean in DeepStream?
- What changes when deploying on Jetson instead of a desktop GPU?
- What constraints determine whether a layer can run on DLA?
- How do you expose a C++ TensorRT engine to Python safely?

After `22_llm_inference_intro`, you should be able to answer:

- What are prefill and decode?
- What is KV cache?
- Why is LLM decoding often memory-bandwidth bound?
- How is LLM batching different from YOLO image batching?
- What problems do TensorRT-LLM, OpenVINO GenAI, vLLM, and llama.cpp try to solve?

After `24_benchmark_report`, you should be able to answer:

- What is the best latency you achieved on this RTX 2060?
- How much faster is FP16 than FP32?
- Did INT8 help on this model and GPU?
- What accuracy trade-off did you observe?
- Which evidence came from single-input tensor alignment, and which came from multi-image detection
  validation?
- What would you change for production deployment?

## Portfolio Story

The final story should be:

I took YOLOv8n from PyTorch to ONNX, validated it, used Polygraphy for single-input ONNX Runtime and TensorRT precision alignment, extended validation to multi-image FP32/FP16/INT8 detection-quality checks, built TensorRT FP32/FP16/INT8 engines, implemented a C++ inference pipeline with RAII-managed TensorRT/CUDA resources, OpenCV preprocessing, YOLO postprocessing, producer-consumer video flow, dynamic batching, and multi-stream video scheduling, measured latency and throughput on RTX 2060, diagnosed bottlenecks with Nsight Systems, handled unsupported operators with graph surgery and a runnable custom TensorRT plugin, compared the same model with OpenVINO CPU inference, ran a DeepStream-style multi-stream path, documented Jetson Orin/Xavier DLA deployment constraints, exposed C++ inference to Python, learned the entry-level LLM inference concepts, and summarized the engineering trade-offs in a benchmark report.
