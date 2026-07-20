# TensorRT Learning Roadmap

This roadmap is designed for an AI algorithm engineer who already understands model training and delivery, but wants to build practical deployment and inference optimization skills for TensorRT, OpenVINO, and C++ inference engineering.

The goal is not to collect demos. The goal is to build a portfolio-quality project that proves you can take a model from PyTorch to a production-like inference pipeline, measure it, optimize it, and explain the trade-offs.

## Target Outcome

The core path and elective tracks are designed so that, after completing the relevant lessons, you
can:

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
- Serve TensorRT models with Triton and measure server-side batching, concurrency, and metrics.
- Measure latency, throughput, memory usage, tensor drift, and task-level accuracy changes.
- Explain performance bottlenecks in CPU preprocessing, GPU inference, memory copies, and synchronization.
- Deploy the same model with OpenVINO for CPU comparison.
- Understand the minimum LLM inference concepts needed for modern deployment interviews.
- Present the project as an interview-ready deployment case study.
- Produce interview-ready evidence at three checkpoints instead of postponing all reporting until the
  end of the course.

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
- Record the TensorRT, CUDA, driver, GPU, and container versions behind every engine and benchmark.
  Serialized engines are deployment artifacts tied to a compatibility context, not portable model
  files to copy blindly between environments.
- Keep the pinned TensorRT 23.10 environment as the reproducible baseline for completed lessons.
  When a lesson needs current APIs such as `IPluginV3`, give that lesson an explicit compatible
  container and document the migration boundary instead of silently changing every earlier result.

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

Use the core path in order. After the core path, choose an elective track from the job descriptions
you are targeting instead of completing every elective sequentially. `23_cpp_interview_katas` runs
alongside the course rather than waiting until the end.

| Path | Sequence | Focus |
| --- | --- | --- |
| Core foundation | `00` through `10` | Environment, C++, CUDA, ONNX, TensorRT, and end-to-end YOLO C++ inference |
| Checkpoint 1 | `10a` | Functional validation, architecture evidence, and an English project explanation |
| Core optimization | `11` through `12` | Nsight diagnosis plus an advanced FP16/INT8 quantization and deployment case study |
| Checkpoint 2 | `12a` | Reproducible precision and performance report; begin targeted applications |
| Core pipeline | `13` through `17` | Queues, dynamic batching, async video, multi-stream scheduling, and CUDA/NPP preprocessing |
| Checkpoint 3 | `17a` | Pipeline load, latency, throughput, and stability report |
| Server elective | `18a`, `21` | Triton serving and C++/Python integration |
| Edge CV elective | `20`, `20a` | DeepStream, GStreamer, Jetson, and DLA |
| Advanced TensorRT elective | `19`, `19a` | Graph surgery and a runnable TensorRT plugin |
| CPU/Intel elective | `18` | OpenVINO CPU inference comparison |
| LLM awareness elective | `22` | Entry-level LLM inference concepts and measurements |
| Ongoing interview practice | `23` | Deployment-relevant C++ exercises tied to completed lessons |
| Final synthesis | `24` | Portfolio case study, packaging, resume evidence, and English presentation |

## Course Plan

This document is the source of truth for planned lessons. A lesson directory is created only when
implementation starts, and that implementation should include its runnable artifact, concise
README, and proportionate verification. Planned lessons do not need README-only placeholder
directories.

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
- Explicit TensorRT and CUDA error propagation
- Failure injection during staged initialization
- Repeated resource construction and destruction

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
- Missing, corrupt, or incompatible engine input produces an explicit error without leaking resources.
- Injected allocation or setup failures release every resource acquired earlier in initialization.
- A repeated create/destroy test exercises runtime, context, buffer, and stream ownership without
  increasing host or device memory use.
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

### `10a_end_to_end_validation_report`

Purpose:

- Turn the first complete inference pipeline into reviewable evidence before starting performance
  optimization.
- Practice explaining the project in English while the architecture and debugging decisions are
  still fresh.

Deliverables:

- `reports/10a_end_to_end_validation.md`
- Environment and dependency table
- PyTorch, ONNX Runtime, and TensorRT comparison using the same controlled input
- Pipeline architecture and ownership notes
- Build, run, and test commands that work from a clean course container
- Per-stage latency baseline without claiming that it is optimized
- One-page English summary and a three-to-five-minute English walkthrough

Acceptance criteria:

- Another engineer can reproduce the end-to-end result from the documented commands.
- The report distinguishes functional correctness from task-level accuracy and performance claims.
- Known limitations and the next measurement questions are explicit.
- Report values come from saved command output or machine-readable results rather than manually
  maintained duplicate numbers.

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

### `12_yolov8_int8_quantization_engineering`

Purpose:

- Treat INT8 quantization as an advanced engineering case study rather than a one-command engine build.
- Follow the technology evolution from legacy TensorRT PTQ to mixed precision and ModelOpt explicit Q/DQ.
- Make the final deployment decision from fixed quality gates and version-matched performance evidence.

Engineering sequence:

1. Create a versioned calibration-data contract.
2. Establish immutable PyTorch, TensorRT FP32, and FP16 reference bundles.
3. Compare legacy Entropy and MinMax calibration with one-variable-at-a-time experiments.
4. Run one complete-detection-head FP16 sensitivity experiment.
5. Export ModelOpt explicit-Q/DQ models in a separate pinned environment.
6. Rebuild FP32 and FP16 references when moving from TensorRT 8.6 to TensorRT 10.
7. Benchmark only quality-passing candidates and retain FP16 when INT8 has no matched benefit.

Topics:

- Fixed 5,000-image COCO train2017 candidate-pool identity
- Independently coverage-selected 3,000-image calibration manifest
- Complete 5,000-image COCO val2017 human-labeled validation split
- Calibration/validation hash-overlap rejection
- Byte-identical calibration and evaluation preprocessing verification
- Predeclared mAP50-95, mAP50, precision, and recall regression thresholds
- Immutable reference bundles and candidate-only evaluation
- TensorRT 8.6 Entropy and MinMax legacy calibrators
- Calibration cache identity and persistent timing caches
- Strict complete-detection-head FP16 precision constraints
- Engine Inspector validation of requested and actual precision
- ModelOpt explicit `QuantizeLinear`/`DequantizeLinear` export
- TensorRT 8.6 FP32-high-precision Q/DQ build
- TensorRT 10.14 native-FP16 strongly typed Q/DQ build
- Runtime-version evidence boundaries
- Matched `trtexec` latency, GPU compute, and throughput measurements
- Reformat and Q/DQ-boundary evidence for slower mixed-precision execution

Recorded result:

- Legacy Entropy: quality FAIL.
- Legacy MinMax: quality FAIL because mAP50 misses the unchanged gate.
- MinMax with the complete detection head in FP16: quality FAIL.
- ModelOpt Q/DQ under TensorRT 8.6: quality PASS.
- ModelOpt native-FP16 Q/DQ under TensorRT 10.14: quality PASS.
- Matched TensorRT 10 throughput: FP16 `636.729 qps`; INT8+FP16 `522.188 qps`.
- Deployment retains FP16 because the quality-passing INT8 candidate is slower.

Acceptance criteria:

- The candidate-pool identity, selected calibration manifest, validation manifest, hashes, and selection method are saved.
- Calibration and validation contain no duplicate image content across splits.
- All 3,000 calibration images produce byte-identical tensors through both production preprocessing paths.
- PyTorch, FP32, and FP16 references are computed once per runtime identity and reused for later candidates.
- TensorRT 8.6 references are rejected for TensorRT 10 candidates.
- Each experiment changes one declared quantization variable and writes distinct artifacts.
- Failed quality candidates are excluded from authoritative deployment performance comparisons.
- Explicit Q/DQ candidates pass or fail the same thresholds used by legacy PTQ.
- Matched performance uses the same TensorRT version, GPU, shapes, warmup, iterations, stream count, and transfer settings.
- The final report explains why passing accuracy is necessary but insufficient for deploying INT8.
- The generated case study and machine-readable summary agree with the raw ignored evidence.


### `12a_precision_performance_report`

Purpose:

- Produce the first application-ready benchmark report instead of waiting for every later elective.
- Demonstrate that precision and speed decisions are supported by reproducible measurements.

Deliverables:

- `reports/12a_precision_performance.md`
- Hardware, software, power-state, warmup, iteration-count, and synchronization methodology
- FP32, FP16, and INT8 latency and throughput tables
- Nsight Systems timeline evidence and bottleneck explanation
- Single-input raw tensor alignment linked to multi-image drift and detection-quality results
- Model, dataset, evaluator, preprocessing, and postprocessing version information
- Calibration and validation dataset manifests with no overlap
- mAP50-95, mAP50, precision, recall, absolute values, backend deltas, and predeclared regression thresholds
- One-page English summary and a three-to-five-minute English benchmark explanation

Acceptance criteria:

- Every reported number has a command, saved result, or generated artifact behind it.
- P50/P90/P99 latency is calculated from individual samples after a documented warmup.
- Accuracy conclusions separate numerical drift from decoded detection-quality changes.
- The report states pass or fail against thresholds that were fixed before the final evaluation run.
- Accuracy tables are generated from evaluator output rather than transcribed manually.
- The report identifies at least one evidence-backed optimization and one rejected or inconclusive
  optimization attempt.
- This checkpoint is strong enough to support targeted applications while later pipeline work
  continues.

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
- Explicit queue `close()` semantics
- Cancellation and exception propagation across worker threads
- Repeated start/stop and bounded stress testing
- ThreadSanitizer for the CPU-only queue and synchronization tests

Acceptance criteria:

- One thread reads images or video frames and pushes them into a bounded thread-safe queue.
- Another thread pops frames and simulates or runs inference.
- The queue has a clear policy when input FPS is higher than inference FPS.
- Closing the queue wakes blocked producers and consumers, rejects new pushes, and lets queued work
  follow the documented drain-or-discard policy.
- Producer and consumer failures are propagated to the owner instead of being lost inside a thread.
- Repeated start/stop and overload tests keep queue depth and memory use bounded.
- The program exits cleanly without deadlock.
- CPU-only queue stress tests complete without ThreadSanitizer findings.
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
- Producer-consumer queue integration
- Async inference
- Double buffering
- CPU/GPU overlap
- Dynamic batching from queued frames
- Frame timestamp tracking
- Dropped-frame statistics
- End-of-stream, invalid-input, and worker-failure handling
- Coordinated cancellation and shutdown
- Explicit overload and frame-dropping policy
- FPS, latency-percentile, queue-depth, and error metrics

Acceptance criteria:

- The program can process a video or camera stream.
- You can report average FPS, P50/P90/P99 latency, and GPU utilization.
- Queue depth and memory remain bounded when input FPS exceeds processing capacity.
- Normal end-of-stream drains or discards queued frames according to the documented policy and exits cleanly.
- Invalid input and an injected worker failure stop the pipeline without deadlock and return an
  explicit nonzero error.
- Dropped-frame, processed-frame, and failure counters remain internally consistent.

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
- Per-stream source failure policy
- Inference-worker failure propagation and coordinated cancellation
- Result-identity validation under out-of-order completion

Acceptance criteria:

- The program can read from at least two video files or camera-like sources.
- Each stream has independent FPS, queue depth, and dropped-frame counters.
- Frames are batched for TensorRT inference when possible.
- Detection results are routed back to the correct stream.
- A result-integrity test proves that every output retains the correct `stream_id` and `frame_id`
  under batching and out-of-order completion.
- An injected source or inference-worker failure follows the documented isolate-or-stop policy and
  leaves no blocked threads.
- Queue depth and memory use remain bounded under sustained overload.
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

### `17a_pipeline_performance_report`

Purpose:

- Prove that the pipeline remains measurable, bounded, and cleanly stoppable under realistic load.
- Compare latency, throughput, fairness, and freshness instead of reporting average FPS alone.

Deliverables:

- `reports/17a_pipeline_performance.md`
- Single-stream and multi-stream architecture diagrams
- Queue depth, batching timeout, scheduling, backpressure, and dropped-frame policies
- Total and per-stream throughput plus P50/P90/P99 end-to-end latency
- CPU OpenCV and CUDA/NPP preprocessing correctness and performance comparison
- GPU utilization, memory use, queue depth, and dropped/stale-frame metrics
- Host RSS and device-memory measurements at the start, peak, and end of the run
- A soak test of at least 30 minutes under a documented workload
- At least 100 repeated pipeline start/stop cycles
- A fault-injection matrix covering invalid input, source failure, worker failure, and shutdown during load
- AddressSanitizer and ThreadSanitizer results for applicable CPU-only modules
- Targeted `compute-sanitizer` results for CUDA memory and kernel smoke tests
- Evidence of graceful shutdown under normal and failure paths
- One-page English summary and a three-to-five-minute English pipeline explanation

Acceptance criteria:

- Frame timestamps cover capture-to-result latency rather than inference latency alone.
- The report explains the tested overload policy and demonstrates that memory and queue growth are
  bounded.
- Results show the trade-off between batch efficiency, per-stream fairness, and real-time freshness.
- The pipeline exits without deadlock when input ends, a producer fails, or shutdown is requested.
- Host RSS and device memory do not show unexplained monotonic growth across the soak and repeated
  start/stop tests.
- Applicable sanitizer runs finish without unresolved memory, race, or CUDA access errors.
- Injected failures return explicit errors and match the documented isolate-or-stop policy.
- Benchmark results are generated from saved measurements and extend rather than duplicate the
  earlier precision report.

## Elective Tracks

Choose electives from target job descriptions. The lesson numbers are stable identifiers, not a
requirement to complete these lessons in numerical order.

### `18_openvino_yolov8`

Track: CPU and Intel deployment.

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

### `18a_triton_inference_server`

Track: server inference and AI platform roles.

Purpose:

- Serve the TensorRT model through a standard inference server and measure behavior under concurrent
  client load.

Topics:

- Triton model repository and model configuration
- TensorRT backend
- HTTP and gRPC clients
- Dynamic batching and queue delay
- Model instances and GPU resource trade-offs
- Input/output shape and datatype validation
- Prometheus metrics
- Concurrency Analyzer or an equivalent repeatable load test
- Client latency versus server compute and queue latency
- Model versioning and controlled rollout concepts

Acceptance criteria:

- Triton loads the TensorRT model from a reproducible model repository.
- A client sends real preprocessed inputs and validates structured outputs.
- A benchmark compares at least two concurrency levels and dynamic batching configurations.
- The report includes throughput, client P50/P90/P99 latency, server queue time, and GPU utilization.
- You can explain when Triton dynamic batching helps and when its queue delay violates a latency
  target.

### `19_onnx_graph_surgery_plugin`

Track: advanced TensorRT and unsupported-operator deployment.

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
- TensorRT `IPluginV3` capability interfaces and plugin creator responsibilities
- Plugin serialization, deserialization, and resource ownership
- Legacy `IPluginV2DynamicExt` concepts when maintaining TensorRT 8.x deployments

Acceptance criteria:

- You can explain the three-level strategy: model rewrite, ONNX graph surgery, TensorRT plugin.
- You can edit a small ONNX graph with GraphSurgeon.
- You can describe plugin registration, build-time shape/type negotiation, runtime `enqueue`,
  serialization, and engine deserialization.
- You can explain why a current `IPluginV3` implementation and a legacy TensorRT 8.x plugin use
  different interfaces.

### `19a_custom_tensorrt_plugin`

Track: advanced TensorRT and unsupported-operator deployment.

Purpose:

- Build one runnable custom TensorRT plugin instead of only describing the plugin strategy.

Why it matters:

- Custom plugin experience is a strong signal for roles that deploy non-standard CV, medical, industrial, or research models.
- A small complete plugin demonstrates C++ ABI awareness, CUDA kernel integration, TensorRT lifecycle knowledge, and numerical validation.

Suggested plugin scope:

- Implement a compact operator such as `ScaleShift`, `Clip`, or `CustomNormalize`.
- Keep the operator simple enough that plugin mechanics, serialization, and validation remain the teaching focus.

Topics:

- `IPluginV3`, its core/build/runtime capability interfaces, and resource interfaces where needed
- Plugin creator registration and field-based serialization
- CUDA kernel launch from `enqueue`
- Dynamic shape and data type handling
- Plugin field collection and parameters
- Serialization and deserialization
- Building a plugin shared library
- Loading plugins with `trtexec --plugins`
- Loading plugins from C++ runtime code
- ONNX GraphSurgeon replacement with a plugin node
- Polygraphy or ONNX Runtime reference comparison
- A short TensorRT 8.x-to-current-plugin migration note for the pinned legacy course environment

Acceptance criteria:

- A plugin shared library builds with CMake.
- `trtexec` can load the plugin library and build an engine containing the plugin layer.
- A small C++ or Python runtime example loads the plugin-backed engine and runs inference.
- The plugin output is numerically checked against a CPU/Python reference.
- You can explain the plugin lifecycle from registration to `enqueue` to engine deserialization.

### `20_deepstream_gstreamer_multistream`

Track: edge CV and multi-stream video analytics.

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

Track: edge CV and embedded NVIDIA deployment.

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

Track: server inference integration and reusable deployment libraries.

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

Track: LLM inference awareness for general deployment interviews.

Purpose:

- Build an entry-level understanding of LLM inference so deployment interviews do not stop at CNN/CV models.

Scope:

- This is not the main project line.
- The goal is to understand core concepts and run a small local example, not to build a full LLM serving stack.

Topics:

- Tokenization
- Transformer attention data flow at the level needed to reason about inference memory
- Prefill and decode
- KV cache
- Batch size versus sequence length
- Batch size versus concurrent-request semantics in the selected backend
- Throughput versus first-token latency
- FP16, INT8, INT4, and weight-only quantization
- ONNX Runtime, OpenVINO GenAI, TensorRT-LLM, vLLM, and llama.cpp at a high level
- CPU/GPU memory limits
- Pinned model revision, tokenizer, quantization format, runtime, and benchmark configuration
- Warmup, repeated measurements, and fixed output length
- TTFT, time per output token, prefill throughput, decode throughput, and total tokens per second
- Peak GPU memory plus model-weight and KV-cache memory estimates
- Controlled input-length and batch-or-concurrency experiment matrix

Acceptance criteria:

- You can run one small local model rather than only reading an inference example.
- The benchmark records the exact model revision, tokenizer, quantization format, backend, hardware,
  warmup, repetition count, input length, and requested output length.
- With output length held constant, results compare at least two input lengths and at least two batch
  sizes or concurrent-request levels supported by the selected backend.
- The result table reports TTFT, time per output token, prefill throughput, decode throughput, total
  tokens per second, and peak GPU memory from repeated post-warmup measurements.
- The report separates model-weight memory from an estimated KV-cache contribution and records any
  configuration that cannot run within available memory.
- You can explain why LLM inference bottlenecks differ from YOLO inference.
- You can explain KV cache and why decode is often memory-bandwidth bound.
- You can explain how input length and batch or concurrency changed latency, throughput, and memory
  use in the measured results.
- You know when to mention TensorRT-LLM, OpenVINO GenAI, vLLM, or llama.cpp in interviews.

## Ongoing Interview Practice

### `23_cpp_interview_katas`

Purpose:

- Prepare for practical C++ interview questions related to CV deployment instead of generic puzzle-style questions.
- Practice each group after its related core lesson instead of postponing all exercises until the
  end.

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

Suggested timing:

- After `03`: letterbox mapping and HWC-to-CHW.
- After `07`: RAII wrappers and move-only resource ownership.
- After `10`: IoU, NMS, and Top-K.
- After `13`: bounded queues and ring buffers.

## Final Synthesis

### `24_final_portfolio_case_study`

Purpose:

- Convert the checkpoint reports and selected electives into one concise interview case study.

Deliverables:

- `reports/24_final_portfolio_case_study.md`
- Links to the `10a`, `12a`, and `17a` evidence instead of copying their complete contents
- Concise latency, throughput, accuracy, and pipeline summary tables
- Environment table
- Test evidence table
- CI/build notes
- Production Dockerfile
- Development image versus runtime image size comparison
- Bottleneck analysis
- Future work
- Resume bullets, a five-minute English presentation, and a longer technical interview walkthrough

Acceptance criteria:

- A recruiter or interviewer can understand the project in five minutes.
- You can defend every number in the report.
- You can explain why a single-input Polygraphy pass is useful but not sufficient for release
  approval.
- The final report points to the reusable module structure and the tests that protect core
  preprocessing, postprocessing, and resource-management behavior.
- CI or a documented local equivalent configures, builds, and runs the available tests.
- A multi-stage Docker build produces a runtime image that contains only the files needed to run inference.
- The case study clearly identifies which elective track was completed and why it matches the target
  role.

## Suggested Lesson Routine

For each implementation lesson:

1. Read the API or tool documentation.
2. Implement the smallest runnable version.
3. Add command examples to the lesson README.
4. Add timing or correctness checks.
5. Write three notes: what worked, what failed, what changed your mental model.

For each report checkpoint, collect saved measurements from the preceding lessons, regenerate the
tables, record limitations, and rehearse the English explanation. A report checkpoint should not
introduce an unrelated deployment feature.

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
- How do you inject an initialization failure without depending on a real out-of-memory event?
- What evidence distinguishes correct RAII ownership from a long-run leak claim?

After `10_yolov8_trt_cpp`, you should be able to answer:

- How does YOLO preprocessing work?
- How do you allocate and bind TensorRT input/output buffers?
- Where are the main latency costs?
- What parts run on CPU and what parts run on GPU?
- How do you validate TensorRT output against PyTorch output?

After `10a_end_to_end_validation_report`, you should be able to answer:

- Can another engineer reproduce the pipeline from a clean environment?
- Which evidence establishes functional correctness, and which claims are still unproven?
- How are TensorRT, CUDA, and OpenCV resources owned across the pipeline?

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

After `12a_precision_performance_report` and `17a_pipeline_performance_report`, you should be able to answer:

- How were warmup, synchronization, sample count, and percentile latency defined?
- How do you prevent calibration/validation leakage and enforce a predeclared accuracy gate?
- Why can raw tensor drift pass while task-level mAP still regresses, or the reverse?
- What is the difference between inference latency and capture-to-result latency?
- Which bottleneck was proven by evidence, and which optimization did not help?
- How does overload affect queue depth, dropped frames, fairness, and freshness?
- Which failures were injected, which sanitizers were applicable, and what does a 30-minute soak test
  still not prove?

After `18a_triton_inference_server` and `21_cpp_shared_library_python_binding`, you should be able to answer:

- How does Triton discover, configure, and version a TensorRT model?
- How do dynamic batching, queue delay, concurrency, and model instances interact?
- Which latency is measured by the client, and which components are measured by the server?
- How do you expose a C++ TensorRT engine to Python safely?

After `20_deepstream_gstreamer_multistream` and `20a_jetson_orin_xavier_dla_deployment`, you should be able to answer:

- What is a GStreamer pipeline?
- What does `nvstreammux` do?
- What does zero-copy mean in DeepStream?
- What changes when deploying on Jetson instead of a desktop GPU?
- What constraints determine whether a layer can run on DLA?

After `22_llm_inference_intro`, you should be able to answer:

- What are prefill and decode?
- What is KV cache?
- What are TTFT and time per output token, and why must output length be controlled in a comparison?
- Why is LLM decoding often memory-bandwidth bound?
- How is LLM batching different from YOLO image batching?
- How did input length and batch or concurrency affect measured throughput, latency, and memory use?
- What problems do TensorRT-LLM, OpenVINO GenAI, vLLM, and llama.cpp try to solve?

After `24_final_portfolio_case_study`, you should be able to answer:

- What is the best latency you achieved on this RTX 2060?
- How much faster is FP16 than FP32?
- Did INT8 help on this model and GPU?
- What accuracy trade-off did you observe?
- Which evidence came from single-input tensor alignment, and which came from multi-image detection
  validation?
- What would you change for production deployment?

## Portfolio Story

The final story should lead with the core evidence:

I took YOLOv8n from PyTorch to ONNX, aligned ONNX Runtime and TensorRT outputs, built and validated
FP32/FP16/INT8 engines, implemented a C++17 inference pipeline with explicit TensorRT/CUDA resource
ownership, and extended it with bounded queues, dynamic batching, asynchronous execution, and
multi-stream scheduling. I measured task-level accuracy, latency percentiles, throughput, GPU
utilization, and dropped-frame behavior, then used Nsight Systems evidence to explain the resulting
engineering trade-offs.

Add only the elective evidence that was actually implemented and verified. For example, a server
candidate can add Triton concurrency and dynamic-batching results, while an edge CV candidate can
add DeepStream and Jetson evidence. Do not present every roadmap elective as completed work.
