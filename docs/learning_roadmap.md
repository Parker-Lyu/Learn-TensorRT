# TensorRT Learning Roadmap

This roadmap is designed for an AI algorithm engineer who already understands model training and
delivery, but wants to build practical deployment and inference optimization skills for TensorRT,
OpenVINO, and C++ inference engineering.

The goal is not to collect demos. The goal is to build a portfolio-quality project that proves you
can take a model from PyTorch to a production-like inference pipeline, measure it, optimize it, and
explain the trade-offs.

## Course Baseline

The roadmap uses one development environment for the complete core path:

- `nvcr.io/nvidia/pytorch:25.11-py3`
- TensorRT 10.14 (`10.14.1.48` in the pinned image)
- CUDA Toolkit 13.0
- ISO C++17 for host C++ and CUDA C++ targets

PyTorch export and ModelOpt explicit-Q/DQ work run in the same environment as TensorRT C++ builds
and validation. A derived development image may add OpenCV, ONNX Runtime, Ultralytics, and test
tools without replacing the base CUDA, TensorRT, or PyTorch stack.

Build all TensorRT engines, timing caches, reference outputs, reports, and benchmark evidence with
the baseline above. Record enough environment metadata for a third party to reproduce each result.

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

Engineering capabilities developed through the course:

- Reusable logic is organized behind small C++ APIs, with headers and source files separated when a
  lesson grows beyond one concept.
- Reusable components become CMake library targets linked by small lesson executables.
- Focused tests protect reusable algorithms and resource wrappers once meaningful edge cases appear.
- C++, CUDA, TensorRT, PyTorch export, ModelOpt, and validation share the development environment
  derived from `nvcr.io/nvidia/pytorch:25.11-py3`. Runtime packaging is introduced near the final
  portfolio stage, after the inference pipeline is complete.
- Treat Unified Memory and zero-copy-style paths as performance trade-offs to measure, not as magic
  shortcuts. `cudaMallocManaged` simplifies ownership but can still page migrate on discrete GPUs.
- Record the TensorRT, CUDA, driver, GPU, and container versions behind every engine and benchmark.
  Serialized engines are deployment artifacts tied to a compatibility context, not portable model
  files to copy blindly between environments.
- TensorRT 10.14, CUDA Toolkit 13.0, and ISO C++17 remain the reproducible baseline for every core
  lesson. Electives that study another platform or runtime make that portability boundary explicit.

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

This final structure is a destination rather than the starting point for lesson 01. Each implemented
lesson provides one runnable artifact and one concise README.

## Learning Flow

Use the core path in order. After the core path, choose an elective track from the job descriptions
you are targeting instead of completing every elective sequentially. `31_cpp_interview_katas` runs
alongside the course rather than waiting until the end.

| Path | Sequence | Focus |
| --- | --- | --- |
| Core foundation | `00` through `11` | Environment, C++, CUDA, ONNX, TensorRT, and end-to-end YOLO C++ inference |
| Checkpoint 1 | `12` | Functional validation, architecture evidence, and an English project explanation |
| Core optimization | `13` through `14` | Nsight diagnosis plus an advanced FP16/INT8 quantization and deployment case study |
| Checkpoint 2 | `15` | Reproducible precision and performance report; begin targeted applications |
| Core pipeline | `16` through `21` | Queues, dynamic batching, async/multi-stream scheduling, CUDA/NPP preprocessing, and integrated TensorRT execution |
| Checkpoint 3 | `22` | Integrated pipeline load, latency, throughput, reliability, and stability report |
| Server elective | `24`, `29` | Triton serving and C++/Python integration |
| Edge CV elective | `27`, `28` | DeepStream, GStreamer, Jetson, and DLA |
| Advanced TensorRT elective | `25`, `26` | Graph surgery and a runnable TensorRT plugin |
| CPU/Intel elective | `23` | OpenVINO CPU inference comparison |
| LLM awareness elective | `30` | Entry-level LLM inference concepts and measurements |
| Ongoing interview practice | `31` | Deployment-relevant C++ exercises tied to completed lessons |
| Final synthesis | `32` | Portfolio case study, packaging, resume evidence, and English presentation |

## Course Plan

This document is the source of truth for the planned course sequence. Each implemented lesson
provides a runnable artifact, concise documentation, and verification proportionate to its scope.

Every lesson description uses the same five-part contract:

- **Purpose** defines the problem, motivation, and lesson boundary.
- **Learning outcomes** state what a learner should be able to implement, explain, compare, or
  diagnose after completing the lesson.
- **Topics** list the APIs, concepts, and engineering techniques covered by the lesson.
- **Deliverables** identify the runnable code, tests, scripts, reports, or other reviewable artifacts.
- **Acceptance criteria** define observable evidence that the lesson is complete.

Elective lessons also carry a `Track` label. A lesson may add concise design notes when a sequence,
reference architecture, or constrained implementation scope is part of the lesson itself. The
roadmap defines the course contract; each lesson README provides aligned prerequisites, execution,
output, and learner-checkpoint instructions, plus setup, build, and test instructions when they
apply.

The deliverables below describe artifacts that exist in the repository. Acceptance criteria are
completion gates, not blanket claims that every GPU-, server-, or target-hardware-dependent run has
already passed. [`coverage_matrix.md`](coverage_matrix.md) records the current verification boundary
for partially completed runtime and hardware tracks.

### `00_environment_check`

Purpose:

- Record the local machine, driver, CUDA, TensorRT, Python, compiler, and OpenCV versions.
- Keep reproducible commands for checking the environment.
- Verify the single `nvcr.io/nvidia/pytorch:25.11-py3` development environment before later lessons.

Learning outcomes:

- Verify that the pinned development container exposes the required GPU, CUDA, TensorRT, compiler,
  Python, and course dependencies.
- Explain the compatibility boundary between the host driver and the container-provided CUDA and TensorRT stack.

Topics:

- Host driver and container compatibility
- CUDA Toolkit and TensorRT version checks
- Compiler, CMake, Python, PyTorch, ModelOpt, ONNX, and OpenCV checks
- Reproducible development-container entry and diagnostics

Deliverables:

- `check_env.sh` environment verifier
- `agent_env_setup.md` container-preparation guide
- `README.md` with container entry and verification commands

Acceptance criteria:

- Running `check_env.sh` in the pinned development container ends with the documented pass status.
- The captured output identifies the GPU, driver, container, compiler, CUDA, TensorRT, Python, and
  required course dependencies.
- The report identifies TensorRT 10.14, CUDA Toolkit 13.0, and ISO C++17 as the course baseline.

### `01_hello_world`

Purpose:

- Build confidence with C++17 and CMake.

Learning outcomes:

- Configure, build, and run a minimal ISO C++17 executable with CMake.
- Explain target creation, compile-feature selection, and the effect of disabling compiler extensions.

Topics:

- C++17 executable
- Target-based `CMakeLists.txt`
- Configure, build, and run workflow
- Required standard and disabled compiler extensions

Deliverables:

- `hello_world` CMake executable
- `CMakeLists.txt` enforcing ISO C++17
- `README.md` with clean build and run commands

Acceptance criteria:

- A clean CMake configure and build produces the `hello_world` executable.
- Running the executable prints the documented success message.
- The target requires C++17 with compiler extensions disabled.

### `02_opencv_read_image_info`

Purpose:

- Learn image loading, `cv::Mat` metadata, and basic OpenCV project setup.

Learning outcomes:

- Load an image with OpenCV and interpret its dimensions, channel count, depth, and matrix type.
- Link an OpenCV executable with target-based CMake and report invalid input explicitly.

Topics:

- OpenCV image loading
- `cv::Mat` dimensions, channels, depth, and type
- OpenCV linking with CMake
- Invalid path and command-line validation

Deliverables:

- `opencv_read_image_info` CMake executable
- Image metadata and explicit error output
- `README.md` with default and custom-image runs

Acceptance criteria:

- The executable loads `assets/img.jpeg` by default and prints its width, height, channel count, and OpenCV type.
- A custom readable image path is accepted.
- Invalid arguments or an unreadable image return a nonzero status with an explicit error.

### `03_opencv_preprocess`

Purpose:

- Implement YOLO-style preprocessing outside the model framework.
- Start separating reusable preprocessing logic from the executable entry point.

Learning outcomes:

- Implement reusable YOLO letterbox, color conversion, normalization, and HWC-to-CHW preprocessing.
- Explain the input tensor contract and map detection coordinates back to the original image.
- Validate preprocessing helpers against invalid inputs and boundary cases.

Topics:

- Resize
- Letterbox
- BGR/RGB conversion
- Normalization
- HWC to CHW
- `float32` host buffer
- Batch buffer layout
- TensorRT 10 explicit batch and model-specific input tensor contracts

Deliverables:

- `opencv_preprocess` executable and reusable `preprocess` library
- `preprocess_tests` focused test target
- Saved NCHW tensor, preview, and letterbox debug image under `outputs/`

Acceptance criteria:

- Given an input image, the program writes a preprocessed tensor or debug image.
- Letterbox and coordinate-mapping helpers validate invalid inputs and are easy to test from a
  focused test target.

### `04_cuda_memory_stream`

Purpose:

- Learn the CUDA concepts needed for TensorRT inference code.

Learning outcomes:

- Implement explicit-copy, mapped-pinned-memory, and Unified Memory variants of an inference-like CUDA flow.
- Explain ownership, synchronization, transfer, page-migration, and latency trade-offs for each memory strategy.
- Measure GPU work with CUDA events without introducing unnecessary synchronization.

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

Deliverables:

- `cuda_memory_stream` executable with focused CUDA memory-flow helpers
- Explicit-copy, mapped-memory, and managed-memory execution modes
- Per-mode correctness and CUDA-event timing output

Acceptance criteria:

- Every documented memory mode completes its inference-like CUDA flow and passes the correctness comparison.
- The output reports CUDA-event timings and the synchronization points used by each mode.
- CUDA allocations and streams are released on normal and failure paths.

### `05_torch_to_onnx`

Purpose:

- Export YOLOv8n to ONNX and validate the exported graph.

Learning outcomes:

- Export static- and dynamic-shape YOLOv8n ONNX models from the pinned PyTorch stack.
- Inspect model tensor contracts and validate ONNX Runtime raw outputs against PyTorch on the same input.
- Diagnose numerical mismatches using saved tensors and reproducible tolerance evidence.

Topics:

- Ultralytics YOLOv8n export
- ONNX opset
- Static and dynamic shapes
- ONNX Runtime validation
- `onnxsim`
- Netron graph inspection

Deliverables:

- `export_yolov8_onnx.py`, `inspect_onnx.py`, and `validate_onnx_runtime.py`
- Generated static and dynamic ONNX models in the ignored output directory
- Saved input, raw outputs, graph inspection, and validation report artifacts

Acceptance criteria:

- `yolov8n.onnx` is generated.
- ONNX Runtime output is numerically close to PyTorch output for the same image.
- The inspection report records the model input and output tensor names, shapes, and data types.

### `06_trtexec_engine`

Purpose:

- Learn TensorRT engine construction before writing C++ code.

Learning outcomes:

- Build strict FP32, legacy weakly typed FP16, and modern strongly typed FP16 static/dynamic
  engines with `trtexec`, understanding why `--fp16` is retained only for compatibility.
- Generate explicit mixed-precision FP16 ONNX graphs with ModelOpt AutoCast and pass a reproducible
  ONNX Runtime raw-output conversion gate before TensorRT consumes them.
- Interpret latency, throughput, memory, layer-profile, timing-cache, and environment evidence.
- Explain why serialized engines and timing caches belong to a recorded compatibility context.

Topics:

- `trtexec --onnx`
- Strict FP32 reference with TF32 disabled, plus an FP16 comparison engine
- Runtime/CUDA/GPU/driver/container metadata for benchmark evidence
- Static shape
- Dynamic shape profiles
- Workspace memory
- Layer profiling
- Engine serialization

Deliverables:

- `build_and_benchmark.py` engine-build and benchmark driver
- `summarize_results.py` evidence summarizer
- Ignored engines, timing cache, logs, profiles, timing samples, manifest, and benchmark summary

Acceptance criteria:

- The build driver produces strict static FP32, static FP16, and dynamic FP16 engines.
- Saved benchmark evidence includes latency, throughput, memory, layer profiles, timing samples,
  and environment identity.
- The generated summary compares FP32 and FP16 under the documented matched conditions.

### `07_polygraphy_precision_alignment`

Purpose:

- Learn a repeatable single-input precision-debug workflow when ONNX Runtime and TensorRT outputs
  disagree.
- Real deployment work is not finished when an engine builds successfully.
- Senior candidates should be able to prove where numerical drift starts instead of guessing
  whether preprocessing, export, precision mode, or TensorRT parsing caused the issue.
- A one-image tensor comparison is a debugging gate, not a dataset-level release criterion. Later
  lessons extend it into multi-image drift statistics and decoded detection-quality comparison.

Learning outcomes:

- Run a controlled ONNX Runtime versus TensorRT raw-output comparison with Polygraphy.
- Localize numerical drift to input preparation, export, precision selection, or TensorRT conversion.
- Distinguish a single-input debugging gate from dataset-level detection validation.

Topics:

- Polygraphy model inspection
- ONNX Runtime versus TensorRT comparison
- Saving and comparing one controlled input tensor and its raw model output
- Layerwise or tensorwise debug workflow
- FP32 and FP16 drift analysis for a controlled sample
- Tolerance selection for deployment reports
- Reproducible command logs for interview discussion

Deliverables:

- `align_precision.py` controlled comparison workflow
- Saved Polygraphy logs and backend outputs
- `precision_report.json` and generated precision-alignment note

Acceptance criteria:

- The workflow runs Polygraphy against the YOLO ONNX model and a TensorRT engine with the same saved input tensor.
- Saved evidence includes both backend outputs, mismatch statistics, command logs, and environment identity.
- The generated alignment note identifies the likely source of any mismatch and states that
  multi-image detection validation still follows.

### `08_tensorrt_raii_resource`

Purpose:

- Make TensorRT C++ code exception-safe and long-running-service friendly.
- Industrial camera systems and edge inference services often run 24x7.
- A small host memory leak, CUDA memory leak, or forgotten TensorRT object can become a production incident.
- RAII proves that resources are released even when early returns or exceptions happen.

Learning outcomes:

- Design move-only RAII wrappers for TensorRT objects, CUDA buffers, and CUDA streams.
- Propagate initialization errors without leaking resources acquired earlier.
- Explain how explicit ownership supports long-running inference services.

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

Deliverables:

- `tensorrt_raii_lib` reusable C++ library
- `tensorrt_raii_resource` executable
- `tensorrt_raii_config_tests` focused configuration and failure-path tests

Design notes:

**Example targets:**

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

### `09_tensorrt_cpp_basic`

Purpose:

- Write a minimal TensorRT C++ runtime program.

Learning outcomes:

- Build or deserialize a TensorRT engine and execute one inference through TensorRT 10 name-based APIs.
- Validate tensor names, shapes, data types, formats, and buffer sizes before enqueueing work.
- Explain the lifetime relationships among builder, parser, runtime, engine, context, buffers, and stream.

Topics:

- Logger
- Builder and ONNX parser
- TensorRT 10 strongly typed network creation
- Strict FP32 by default and optional TF32 kernel math
- Strictly verified timing-cache reuse
- Runtime creation and engine deserialization
- Execution context
- Name-based tensor metadata and address binding
- TensorRT 10 IO data-type and format validation
- Device buffer allocation and `enqueueV3`

Deliverables:

- `tensorrt_cpp_basic_lib` reusable C++ library
- `tensorrt_cpp_basic` engine-build/load and inference executable
- Generated engine and timing cache in the ignored output directory

Acceptance criteria:

- A C++ program loads a TensorRT engine and runs one inference with dummy or real input.
- Builder, parser, engine, runtime, context, buffer, and stream lifetimes are clear.
- Strongly typed network creation and explicit tensor precision are used consistently.

### `10_yolov8_trt_python`

Purpose:

- Build a fast debugging reference before the full C++ implementation.

Learning outcomes:

- Run a TensorRT 10 YOLOv8 inference pipeline from Python with real image input.
- Implement NumPy preprocessing, output decoding, NMS, and saved visualization.
- Compare structured detections with the PyTorch or Ultralytics reference.

Topics:

- TensorRT 10 name-based Python runtime and `execute_async_v3`
- NumPy preprocessing
- Output decoding
- NMS
- Visualization

Deliverables:

- `infer_yolov8_trt.py` inference CLI
- Saved detection JSON and annotated image under `outputs/`
- Documented TensorRT engine prerequisite

Acceptance criteria:

- The Python pipeline produces boxes on a test image.
- The output is close to the PyTorch or Ultralytics reference.

### `11_yolov8_trt_cpp`

Purpose:

- Build the main portfolio artifact: end-to-end YOLOv8n TensorRT C++ inference.
- Begin converging lesson code into reusable preprocessing, inference, and postprocessing modules.

Learning outcomes:

- Assemble reusable C++ preprocessing, TensorRT inference, postprocessing, and visualization components.
- Run end-to-end YOLOv8 inference and attribute latency to individual pipeline stages.
- Test coordinate mapping, decoding, NMS, and invalid-input behavior.

Topics:

- OpenCV preprocessing
- TensorRT runtime
- Reusable pinned-host and device CUDA buffers
- TensorRT 10 name-based IO APIs and `enqueueV3`
- Header-only NVTX3 integration for CUDA 13
- YOLO decode
- NMS
- Coordinate scaling
- Visualization
- CLI arguments
- Library targets for reusable components
- Focused tests for preprocessing and postprocessing edge cases

Deliverables:

- Reusable preprocessing, TensorRT runner, postprocessing, and visualization libraries
- `yolov8_trt_cpp` command-line executable
- `yolov8_cpp_tests` focused test target and saved inference outputs

Acceptance criteria:

- The program accepts an image path and engine path.
- It saves an output image with detection boxes.
- It reports preprocessing, inference, postprocessing, and total latency.
- Reusable preprocessing, inference, and postprocessing code is not trapped inside `main`.
- Focused tests cover representative invalid input and boundary cases.

### `12_end_to_end_validation_report`

Purpose:

- Turn the first complete inference pipeline into reviewable evidence before starting performance
  optimization.
- Practice explaining the project in English while the architecture and debugging decisions are
  still fresh.

Learning outcomes:

- Combine PyTorch, ONNX Runtime, TensorRT, and C++ evidence for one controlled input and model identity.
- Explain which evidence establishes functional correctness and which accuracy or performance claims remain unproven.
- Present the architecture, ownership decisions, limitations, and baseline results in a reproducible report.

Topics:

- Controlled evidence identity across PyTorch, ONNX Runtime, TensorRT, and C++
- Machine-readable report generation
- Resource-ownership and architecture notes
- Functional-correctness versus accuracy and performance boundaries

Deliverables:

- `generate_report.py` evidence validator and report generator
- `outputs/evidence.json` machine-readable evidence
- `reports/12_end_to_end_validation.md` generated checkpoint report

Acceptance criteria:

- Another engineer can reproduce the end-to-end result from the documented commands.
- The report distinguishes functional correctness from task-level accuracy and performance claims.
- Known limitations and the next measurement questions are explicit.
- Report values come from saved command output or machine-readable results rather than manually
  maintained duplicate numbers.

### `13_nsight_performance_diagnosis`

Purpose:

- Replace guesswork with timeline-based performance diagnosis.
- Profiling immediately after the first C++ pipeline gives you a baseline before optimization.
- High-end deployment roles expect evidence: latency tables, profiler traces, and bottleneck explanations.

Learning outcomes:

- Capture and read an Nsight Systems timeline for the C++ YOLO pipeline.
- Diagnose CPU preprocessing, transfer, synchronization, or GPU-starvation bottlenecks from evidence.
- Compare an optimization with matched before-and-after measurements rather than intuition.

Topics:

- `trtexec` baseline
- Nsight Systems command-line capture with `nsys`
- TensorRT/CUDA/GPU/driver/container identity in profiling evidence
- Strict capture, SQLite export, and statistics-generation gates
- Timeline reading
- CPU preprocessing bottleneck
- H2D and D2H copy gaps
- GPU starvation
- CUDA stream overlap verification
- P50/P90/P99 latency reporting

Deliverables:

- `profile_yolov8_cpp.py` strict Nsight Systems capture workflow
- Ignored capture, SQLite, statistics, environment, and summary artifacts
- CPU-only tests for command construction and evidence gates

Acceptance criteria:

- The profiling workflow produces a valid Nsight Systems capture, SQLite export, and generated
  statistics for the C++ YOLO program.
- The saved diagnosis identifies whether the measured interval is CPU-bound, transfer-bound,
  synchronization-bound, or keeping the GPU busy.

### `14_yolov8_int8_quantization_engineering`

Purpose:

- Build a reproducible YOLOv8 INT8 deployment decision with TensorRT 10.14.
- Establish PyTorch FP32/FP16 and TensorRT FP32/FP16 references before quantization.
- Use ModelOpt explicit Q/DQ as the recommended post-training quantization path.
- Keep a predeclared detection-quality gate and benchmark only passing candidates.

Learning outcomes:

- Build matched FP32 and FP16 references before evaluating ModelOpt explicit-Q/DQ INT8.
- Enforce immutable dataset, preprocessing, evaluator, environment, and quality contracts.
- Audit actual TensorRT layer precision and make a deployment decision from saved quality and performance evidence.

Topics:

- Immutable calibration and validation manifests
- Byte-identical preprocessing and evaluator contracts
- PyTorch and TensorRT FP32/FP16 references
- ModelOpt explicit-Q/DQ INT8 export
- TensorRT Engine Inspector precision audit
- Predeclared task-level quality gates and matched benchmarking

Deliverables:

- Versioned experiment, environment, quality, calibration, and dataset contracts
- ModelOpt export, TensorRT build, precision-audit, validation, and benchmark tools
- Reference-bundle, preprocessing-parity, evaluator, manifest, and contract tests
- Concise generated quantization-run summary and canonical performance evidence for Lesson 15
- `docs/reproduction.md` end-to-end reproduction procedure

Design notes:

**Engineering sequence:**

1. Download COCO data and create immutable calibration and validation manifests.
2. Verify byte-identical preprocessing across calibration and evaluation.
3. Evaluate PyTorch FP32, PyTorch FP16, TensorRT FP32, and TensorRT FP16.
4. Export a ModelOpt Q/DQ graph and build a strongly typed TensorRT 10.14 INT8 engine.
5. Inspect actual layer precision and evaluate mAP50-95, mAP50, precision, and recall.
6. Compare matched FP32/FP16/INT8 performance only after the INT8 gate passes.

Acceptance criteria:

- Dataset identities, image hashes, preprocessing contract, and environment identity are recorded.
- Every backend uses the same validation split, postprocessing, and metric implementation.
- INT8 meets both the PyTorch FP32-relative and TensorRT FP16-relative thresholds in the quality contract.
- Engine Inspector evidence confirms the requested Q/DQ precision and records boundary conversions.
- Performance evidence uses the same TensorRT version, GPU, shapes, warmup, iterations, and transfers.
- The final recommendation is based on saved evidence rather than copied benchmark numbers.

### `15_precision_performance_report`

Purpose:

- Produce the first application-ready benchmark report instead of waiting for every later elective.
- Demonstrate that precision and speed decisions are supported by reproducible measurements.

Learning outcomes:

- Generate a reproducible FP32, FP16, and gated INT8 comparison from machine-readable evidence.
- Separate raw tensor drift, detection-quality regression, and runtime performance conclusions.
- Defend the selected deployment precision and rejected alternatives with matched measurements.

Topics:

- Synchronized per-sample latency and percentile methodology
- FP32, FP16, and quality-gated INT8 comparison
- Engine, dataset, evaluator, and environment identity
- Raw-output drift versus detection-quality regression
- Machine-readable evidence and generated report tables

Deliverables:

- Lesson 14 canonical performance evidence consumed without repeating GPU measurements
- `generate_report.py` evidence validator, report generator, and focused tests
- `reports/15_precision_performance.md` generated decision report

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

### `16_cpp_producer_consumer`

Purpose:

- Learn the C++ concurrency pattern behind real camera and video inference systems.
- A camera may produce frames faster than a model can consume them.
- A single `while` loop hides backpressure, latency buildup, frame dropping policy, and shutdown complexity.

Learning outcomes:

- Implement a bounded, closeable producer-consumer queue and an image-pipeline owner.
- Define overload, drain-or-discard, cancellation, failure-propagation, and shutdown behavior.
- Reason about queue size, throughput, latency, memory bounds, and frame dropping.

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

Deliverables:

- Reusable bounded queue and `producer_consumer_pipeline` library
- `cpp_producer_consumer` executable
- Queue, overload, cancellation, failure, and lifecycle tests

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

### `17_dynamic_batching`

Purpose:

- Learn how to use TensorRT batch dimensions and dynamic optimization profiles for multi-image inference.
- Medical imaging, offline inspection, and multi-camera systems often benefit from batching.
- TensorRT deployment code must correctly calculate input and output offsets for `N x C x H x W` buffers.

Learning outcomes:

- Build and run one TensorRT engine across multiple runtime batch sizes.
- Calculate NCHW input/output offsets and set dynamic shapes through an optimization profile.
- Compare batch latency and throughput using matched benchmark conditions.

Topics:

- Static batch versus dynamic batch
- TensorRT optimization profiles
- `minShapes`, `optShapes`, and `maxShapes`
- Runtime input shape setting
- Batched preprocessing buffer layout
- Output offset calculation
- Throughput versus latency trade-off

Deliverables:

- Dynamic-profile engine-build and input-preparation tools
- Reusable dynamic batch runner and CLI
- Batch-layout tests and saved batch benchmark evidence

Acceptance criteria:

- A TensorRT engine is built with a dynamic batch profile, for example `1x3x640x640` to `4x3x640x640`.
- C++ code can run batch size 1, 2, and 4 with the same engine.
- Input and output buffer offsets are calculated explicitly.
- A benchmark compares batch size 1 and batch size 4 latency and throughput.

### `18_async_video_pipeline`

Purpose:

- Move from single-image demo to a single-stream production-like inference loop.

Learning outcomes:

- Implement a bounded asynchronous single-stream pipeline with explicit ownership and cancellation.
- Measure capture-to-result latency, throughput, queue depth, and dropped-frame behavior.
- Handle end-of-stream, invalid input, overload, and worker failure without deadlock.

Topics:

- Video input
- Frame queue
- Producer-consumer queue integration
- Asynchronous inference-like work behind a replaceable backend boundary
- Two in-flight worker slots as a CPU-testable double-buffering model
- Timeout-based micro-batching from queued frames
- Frame timestamp tracking
- Dropped-frame statistics
- End-of-stream, invalid-input, and worker-failure handling
- Coordinated cancellation and shutdown
- Explicit overload and frame-dropping policy
- FPS, latency-percentile, queue-depth, and error metrics

Deliverables:

- Reusable asynchronous single-stream pipeline library
- Runnable video-pipeline executable
- Lifecycle, overload, end-of-stream, and failure-path tests

Acceptance criteria:

- The program processes its synthetic source and accepts a video file or camera source when one is
  available.
- Saved metrics report average FPS, P50/P90/P99 capture-to-result latency, queue peak, processed
  frames, and dropped frames.
- Queue depth and memory remain bounded when input FPS exceeds processing capacity.
- Normal end-of-stream drains or discards queued frames according to the documented policy and exits cleanly.
- Invalid input and an injected worker failure stop the pipeline without deadlock and return an
  explicit nonzero error.
- Dropped-frame, processed-frame, and failure counters remain internally consistent.
- The README identifies `--inference-ms` as simulated work and does not present it as TensorRT or
  GPU evidence; GPU utilization is reported only after integrating and running a real backend.

### `19_multistream_video_pipeline`

Purpose:

- Handle the real production pattern where one process receives frames from multiple cameras or multiple video files.
- Industrial inspection, traffic perception, retail analytics, and autonomous driving rigs usually
  have more than one stream.
- Multi-stream systems need per-stream buffering, global scheduling, batching, overload handling,
  and per-stream metrics.
- A design that works for one video may fail when four cameras have different FPS, resolution, and jitter.

Learning outcomes:

- Schedule and batch frames from multiple independently bounded streams.
- Preserve stream and frame identity through batching and out-of-order result completion.
- Compare fairness, throughput, latency, and freshness under overload and source failures.

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

Deliverables:

- Reusable multi-stream scheduler and pipeline library
- Runnable multi-source executable
- Identity, fairness, overload, shutdown, and failure-policy tests

Design notes:

**Reference architecture:**

- One capture thread per stream, or a small capture thread pool.
- One bounded queue per stream.
- A scheduler that pulls frames from multiple queues.
- A batch assembler that groups frames into `N x C x H x W`.
- One replaceable inference-worker boundary. The current lesson uses deterministic asynchronous
  work so scheduling remains CPU-testable; lesson 17 provides the TensorRT runner for a later
  integration.
- A result dispatcher that sends detections back to the correct stream by `stream_id` and `frame_id`.

Acceptance criteria:

- The program can read from at least two video files or camera-like sources.
- Each stream has independent FPS, queue depth, and dropped-frame counters.
- Frames are assembled into micro-batches for the replaceable inference-worker boundary.
- Results are routed back to the correct stream without assuming completion order.
- A result-integrity test proves that every output retains the correct `stream_id` and `frame_id`
  under batching and out-of-order completion.
- An injected source or inference-worker failure follows the documented isolate-or-stop policy and
  leaves no blocked threads.
- Queue depth and memory use remain bounded under sustained overload.
- The report includes total throughput and per-stream P50/P90/P99 latency.
- The README does not label the deterministic worker delay as TensorRT inference; end-to-end
  TensorRT detection integration remains a separate completion boundary.

### `20_cuda_preprocess_npp`

Purpose:

- Move preprocessing hotspots from CPU OpenCV to GPU-side code when the timeline proves it is useful.

Learning outcomes:

- Move selected preprocessing work to CUDA/NPP and validate it against the OpenCV reference.
- Measure transfer and preprocessing costs separately before claiming an optimization.
- Explain the trade-offs among explicit copies, mapped memory, Unified Memory, and GPU-native decode paths.

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

Deliverables:

- Reusable CPU/CUDA/NPP preprocessing library
- Correctness and benchmark executable
- Focused preprocessing tests and saved timing evidence

Acceptance criteria:

- At least one preprocessing step runs on GPU.
- The result is numerically checked against the OpenCV implementation.
- A benchmark compares CPU preprocessing and GPU/NPP preprocessing.
- Transfer time is measured separately from preprocessing and inference time.

### `21_integrated_tensorrt_video_pipeline`

Purpose:

- Integrate the queue, dynamic batching, asynchronous scheduling, multi-stream identity handling,
  CUDA/NPP preprocessing, TensorRT execution, YOLO postprocessing, and result dispatch developed
  in lessons 16 through 20 into one runnable pipeline.
- Establish a correct, bounded, and observable TensorRT system without making production
  performance or long-duration stability claims. Those claims belong to lesson 22.

Learning outcomes:

- Compose CPU-testable scheduling and identity logic with a real asynchronous CUDA/TensorRT backend.
- Explain why every concurrently submitted batch needs its own execution context, CUDA stream,
  lifecycle completion event, tensor addresses, and reusable buffers.
- Preserve stream/frame identity and letterbox metadata through dynamic batching and completion in
  any order.
- Bound both CPU queues and GPU in-flight work, and implement drain versus abort shutdown semantics.
- Separate host queueing/staging time from CUDA event timing for H2D, preprocessing, TensorRT,
  D2H, and CPU postprocessing.

Topics:

- Repeatable synthetic, image-sequence, and video-file sources; per-source bounded queues.
- Global round-robin/latest-first scheduling and timeout-based micro-batching (batch 1--4).
- Device-view interfaces that compose lesson 20 preprocessing directly into TensorRT input buffers.
- One shared engine and one context/stream/buffer/event set per in-flight slot.
- Slot lifecycle (`Free`, `Reserved`, `Submitted`, `Completing`, `Failed`) and ownership rules.
- Event-driven completion collection, YOLO decode/NMS, coordinate restoration, and identity-safe
  dispatch.
- Explicit accounting for captured, admitted, evicted, submitted, completed, failed, and aborted
  frames; structured per-stage metrics.
- Normal end-of-stream drain and failure abort. Already submitted CUDA work is quiesced, not claimed
  to be cancellable.

Deliverables:

- `integrated_pipeline_core` library for frame metadata, bounded queues, batch assembly, slot state,
  identity dispatch, and metrics; all core logic has CPU-only tests.
- `integrated_tensorrt_backend` library with asynchronous CUDA/NPP preprocessing and TensorRT
  `enqueueV3()` using slot-local resources. It reuses earlier lesson algorithms through explicit
  asynchronous device-view adapters rather than synchronous host round trips.
- `integrated_tensorrt_video_pipeline` executable for repeatable synthetic/image/video sources,
  configurable queue and slot policies, structured detections, and saved annotated samples.

Acceptance criteria:

- A batch-size-1 GPU run performs preprocessing, TensorRT inference, YOLO decoding/NMS,
  coordinate restoration, and writes identity-bearing detections.
- Batch sizes 1, 2, and 4 use the same dynamic-profile engine; empty/oversized/profile-invalid
  shapes and insufficient capacities are rejected before enqueue.
- Two distinct slots can have overlapping submitted work. The normal path does not use
  `cudaDeviceSynchronize()` to serialize each batch; slot reuse waits for its lifecycle event and
  output collection.
- CPU tests force reverse-order completion and prove dispatch uses metadata, not completion order.
  GPU tests prove concurrent slot ownership but do not assume the GPU naturally completes out of order.
- Normal EOS drains accepted work. Abort stops new submissions, discards unsubmitted work, safely
  quiesces submitted slots, joins workers, and releases CUDA/TensorRT resources without deadlock.
- Structured output records stage timings, environment identity, batch distribution, queue peaks,
  explicit overload counters, and capture-to-result latency with documented clock domains.
- Preprocessing and batched detections are compared with CPU/single-image references using stated
  numerical tolerances. This lesson does not claim soak, restart, sanitizer, Nsight, or production
  performance evidence.

### `22_pipeline_performance_report`

Purpose:

- Generate reproducible performance and reliability evidence from the integrated executable in
  lesson 21, while retaining CPU-only evidence from lessons 16--20 as supporting diagnostics.
- Compare latency, throughput, fairness, freshness, batching efficiency, memory bounds, and failure
  behavior instead of reporting average FPS alone.

Learning outcomes:

- Run controlled single- and multi-stream load, overload, fault-injection, soak, restart, and
  sanitizer campaigns against a recorded TensorRT/CUDA environment.
- Explain trade-offs among batch efficiency, queueing latency, fairness, and real-time freshness.
- Produce machine-readable evidence and a generated report that clearly marks unavailable gates.

Topics:

- Integrated pipeline load matrix and capture-to-result latency percentiles.
- Batch-size distribution, queue/drop accounting, host RSS and device-memory sampling.
- Fault injection, normal drain, abort shutdown, repeated restart, sanitizer and optional Nsight
  evidence.
- Report schema, environment provenance, incomplete-gate handling, and reproducibility.

Deliverables:

- `collect_pipeline_evidence.py` invoking lesson 21 and collecting raw load/reliability evidence.
- `generate_report.py`, focused tests, and ignored `reports/22_pipeline_performance.md`.

Acceptance criteria:

- Evidence is generated from saved measurements produced by lesson 21, not simulated worker delay.
- The report documents tested policies and demonstrates bounded queues/in-flight work where measured.
- Latency uses capture timestamps through result dispatch; failures and shutdown outcomes are explicit.
- Formal soak, restart, sanitizer, Nsight, and target-hardware gates are marked incomplete when the
  required environment or duration was not actually run.


## Elective Tracks

Choose electives from target job descriptions. The lesson numbers are stable identifiers, not a
requirement to complete these lessons in numerical order.

### `23_openvino_yolov8`

**Track:** CPU and Intel deployment.

Purpose:

- Compare TensorRT GPU deployment with OpenVINO CPU deployment.

Learning outcomes:

- Run the same YOLO ONNX model through OpenVINO on CPU.
- Measure latency and throughput under a recorded CPU and software environment.
- Explain when an OpenVINO CPU deployment is preferable to TensorRT GPU deployment.

Topics:

- OpenVINO model loading
- CPU inference
- Async infer requests
- FP32/FP16/INT8 where available
- `benchmark_app`

Deliverables:

- `run_openvino.py` CPU inference and measurement CLI
- `generate_comparison.py` TensorRT/OpenVINO comparison generator
- Metric tests and a documented local dependency setup

Acceptance criteria:

- The same ONNX model runs with OpenVINO.
- The generated comparison records OpenVINO CPU and TensorRT GPU latency with their distinct
  hardware and runtime identities.

### `24_triton_inference_server`

**Track:** server inference and AI platform roles.

Purpose:

- Serve the TensorRT model through a standard inference server and measure behavior under concurrent
  client load.

Learning outcomes:

- Prepare a reproducible Triton TensorRT model repository and send validated client requests.
- Measure concurrency, dynamic batching, queue delay, compute time, throughput, and client latency.
- Explain how model instances and batching configuration affect GPU utilization and latency targets.

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

Deliverables:

- `prepare_model_repository.py` reproducible repository generator
- Validated client and load-test tooling
- Metrics utilities, configuration checks, and focused tests

Acceptance criteria:

- Triton loads the TensorRT model from a reproducible model repository.
- A client sends real preprocessed inputs and validates structured outputs.
- A benchmark compares at least two concurrency levels and dynamic batching configurations.
- The report includes throughput, client P50/P90/P99 latency, server queue time, and GPU utilization.

### `25_onnx_graph_surgery_plugin`

**Track:** advanced TensorRT and unsupported-operator deployment.

Purpose:

- Learn the escalation path for unsupported operators and graph conversion failures.
- Real industrial and medical models often contain operators that TensorRT cannot parse directly.
- Senior candidates are expected to know when to change the model, edit the graph, or write a plugin.

Learning outcomes:

- Diagnose an unsupported ONNX operator and choose among model rewrite, graph surgery, and a TensorRT plugin.
- Rewrite and validate a small ONNX graph with ONNX GraphSurgeon.
- Explain the build-time and runtime responsibilities of TensorRT `IPluginV3` capabilities.

Topics:

- Unsupported operator diagnosis
- PyTorch equivalent replacement
- ONNX GraphSurgeon
- Constant folding
- Node replacement and node splitting
- TensorRT plugin design
- TensorRT `IPluginV3` capability interfaces and plugin creator responsibilities
- Plugin serialization, deserialization, and resource ownership

Deliverables:

- Unsupported-operator demo model and diagnosis tool
- GraphSurgeon rewrite and numerical-validation scripts
- Diagnosis tests and an isolated dependency setup

Acceptance criteria:

- The rewrite tool edits the demo ONNX graph with GraphSurgeon and the validator confirms numerical agreement.
- The lesson documentation traces plugin registration, build-time shape/type negotiation, runtime
  `enqueue`, serialization, and engine deserialization.

### `26_custom_tensorrt_plugin`

**Track:** advanced TensorRT and unsupported-operator deployment.

Purpose:

- Build one runnable custom TensorRT plugin instead of only describing the plugin strategy.
- Custom plugin experience is a strong signal for roles that deploy non-standard CV, medical,
  industrial, or research models.
- A small complete plugin demonstrates C++ ABI awareness, CUDA kernel integration, TensorRT
  lifecycle knowledge, and numerical validation.

Learning outcomes:

- Implement, register, build, serialize, deserialize, and execute a TensorRT `IPluginV3` layer.
- Launch a CUDA kernel from plugin `enqueue` while respecting dynamic shape and data-type contracts.
- Validate plugin output numerically against a reference implementation.

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

Deliverables:

- `ScaleShift` TensorRT plugin shared library
- Plugin ONNX model, engine-build workflow, and C++ validator
- CPU-reference numerical comparison

Design notes:

**Suggested plugin scope:**

- Implement a compact operator such as `ScaleShift`, `Clip`, or `CustomNormalize`.
- Keep the operator simple enough that plugin mechanics, serialization, and validation remain the teaching focus.

Acceptance criteria:

- A plugin shared library builds with CMake.
- `trtexec` can load the plugin library and build an engine containing the plugin layer.
- A small C++ or Python runtime example loads the plugin-backed engine and runs inference.
- The plugin output is numerically checked against a CPU/Python reference.

### `27_deepstream_gstreamer_multistream`

**Track:** edge CV and multi-stream video analytics.

Purpose:

- Learn the industrial multi-stream stack used around TensorRT in NVIDIA edge deployments.

Learning outcomes:

- Configure a DeepStream/GStreamer pipeline for two or more sources and a TensorRT engine.
- Explain the roles of source, muxer, inference, tracker, OSD, sink, and NVMM memory.
- Validate generated configuration before running on a compatible DeepStream environment.

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

Deliverables:

- DeepStream application and inference configuration generators
- YOLOv8 DeepStream parser shared library
- Static configuration validation tests and runtime-asset build script

Acceptance criteria:

- A DeepStream sample runs successfully on the local machine or a compatible environment.
- A TensorRT engine is used through DeepStream configuration.
- At least two video streams are processed concurrently.

### `28_jetson_orin_xavier_dla_deployment`

**Track:** edge CV and embedded NVIDIA deployment.

Purpose:

- Understand how TensorRT deployment changes on Jetson Orin/Xavier edge devices, especially when DLA is involved.
- Many CV deployment roles involve edge boxes, robotics, industrial cameras, or embedded NVIDIA
  platforms rather than only desktop GPUs.
- Jetson work requires version discipline because JetPack, CUDA, TensorRT, cuDNN, DeepStream, and
  kernel drivers are tightly coupled.
- This is an edge-deployment extension. It can be documented on x86 first and fully verified later on a Jetson target.

Learning outcomes:

- Plan a reproducible Jetson-native or aarch64 cross-compiled TensorRT deployment.
- Evaluate DLA compatibility, GPU fallback, power mode, clocks, thermals, and platform version coupling.
- Record which validation is possible on x86 and which evidence requires target Jetson hardware.

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

Deliverables:

- Platform check, engine-build, fallback-analysis, and benchmark tools
- Jetson-native build and DLA verification procedure
- CPU-only tool tests for x86 development

Acceptance criteria:

- The lesson documents the target Jetson hardware, JetPack version, TensorRT version, power mode, and clocks.
- The project has a clear native-build path and a cross-compilation checklist for aarch64.
- A YOLO TensorRT engine is attempted with DLA, with unsupported layers and GPU fallback recorded.
- Latency, throughput, memory, and power-mode notes are compared against the desktop GPU baseline
  when hardware is available.
- If no Jetson target is available, the lesson still records the exact commands and expected
  validation steps for future hardware verification.

### `29_cpp_shared_library_python_binding`

**Track:** server inference integration and reusable deployment libraries.

Purpose:

- Package C++ inference code so higher-level Python business logic can call it.
- Production systems often keep the fast inference core in C++ and orchestration/business state in Python.

Learning outcomes:

- Expose the lesson 17 TensorRT runner through a narrow, stable C ABI.
- Manage opaque session ownership, input/output memory, error codes, and exception boundaries safely.
- Call the shared library from Python `ctypes` and validate structured results.

Topics:

- C ABI wrapper
- `.so` dynamic library
- Header design
- Struct-based input/output
- `ctypes`
- `pybind11`
- Ownership across language boundaries
- Error code versus exception boundary

Deliverables:

- `libtrt_inference.so` with a documented C ABI
- `python/trt_ctypes.py` Python client
- ABI, error-boundary, ownership, and integration tests

Acceptance criteria:

- The TensorRT C++ inference class is compiled into a shared library.
- A Python script loads the library and runs inference.
- The exposed API uses simple inputs and structured outputs.

### `30_llm_inference_intro`

**Track:** LLM inference awareness for general deployment interviews.

Purpose:

- Build an entry-level understanding of LLM inference so deployment interviews do not stop at CNN/CV models.
- This is not the main project line.
- The goal is to understand core concepts and run a small local example, not to build a full LLM serving stack.

Learning outcomes:

- Trace tokenization, causal attention, prefill, decode, and KV-cache growth in a small autoregressive model.
- Measure TTFT, time per output token, throughput, and memory across controlled input-length and batch experiments.
- Explain how LLM inference bottlenecks and batching semantics differ from YOLO inference.

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

Deliverables:

- Deterministic inspectable autoregressive Transformer
- Controlled benchmark and report generator
- Correctness tests and generated LLM inference report

Acceptance criteria:

- The committed local model completes prefill and autoregressive decode rather than only presenting pseudocode.
- The benchmark records the exact model revision, tokenizer, quantization format, backend, hardware,
  warmup, repetition count, input length, and requested output length.
- With output length held constant, results compare at least two input lengths and at least two batch
  sizes or concurrent-request levels supported by the selected backend.
- The result table reports TTFT, time per output token, prefill throughput, decode throughput, total
  tokens per second, and peak GPU memory from repeated post-warmup measurements.
- The report separates model-weight memory from an estimated KV-cache contribution and records any
  configuration that cannot run within available memory.

## Ongoing Interview Practice

### `31_cpp_interview_katas`

Purpose:

- Prepare for practical C++ interview questions related to CV deployment instead of generic puzzle-style questions.
- Practice each group after its related core lesson instead of postponing all exercises until the
  end.

Learning outcomes:

- Implement deployment-relevant C++ algorithms and ownership patterns without framework wrappers.
- Explain validation, complexity, boundary behavior, ownership, and synchronization choices.
- Use focused tests to practice each kata after its related course lesson.

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

Deliverables:

- Reusable C++17 kata library and demo executable
- Focused CPU algorithm, queue, ring-buffer, and CUDA ownership tests
- Documented practice timing tied to earlier lessons

Design notes:

**Suggested timing:**

- After `03`: letterbox mapping and HWC-to-CHW.
- After `08`: RAII wrappers and move-only resource ownership.
- After `11`: IoU, NMS, and Top-K.
- After `16`: bounded queues and ring buffers.

Acceptance criteria:

- Each kata has a small C++ implementation and either a focused executable check or Google Test
  coverage.
- Focused executables or tests exercise IoU, NMS, letterbox mapping, and bounded-queue implementations.
- Destructive edge cases are covered for empty inputs, extreme coordinates, overlapping boxes, and
  queue boundary behavior.

## Final Synthesis

### `32_final_portfolio_case_study`

Purpose:

- Convert the checkpoint reports and selected electives into one concise interview case study.

Learning outcomes:

- Synthesize verified checkpoint and elective evidence into a concise deployment case study.
- Defend every reported latency, throughput, accuracy, architecture, and packaging claim.
- Present the project through recruiter-level, resume-level, and technical-interview narratives.

Topics:

- Checkpoint evidence synthesis
- Reusable architecture and test evidence
- Development-versus-runtime packaging
- Reproducible summary tables and platform identity
- Resume bullets and English technical presentation

Deliverables:

- `generate_case_study.py` evidence-driven report generator
- Local verification tools and focused report tests
- Multi-stage runtime `Dockerfile` and engine-delivery helper
- `reports/32_final_portfolio_case_study.md` generated case study

Acceptance criteria:

- A recruiter or interviewer can understand the project in five minutes.
- Every number in the report links to or is generated from saved evidence with a recorded environment identity.
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

After `07_polygraphy_precision_alignment`, you should be able to answer:

- Why can ONNX Runtime and TensorRT produce different outputs?
- How do you compare two backends with the same input tensor?
- How do you decide whether a mismatch is acceptable numerical drift or a deployment bug?
- Why is one-image tensor alignment insufficient for release approval?
- How would you debug the first layer where accuracy starts to diverge?

After `08_tensorrt_raii_resource`, you should be able to answer:

- What is RAII?
- Why are raw TensorRT pointers risky in production code?
- How do you use `std::unique_ptr` with a custom deleter?
- How do you make CUDA buffers exception-safe?
- What happens if engine loading fails halfway through initialization?
- How do you inject an initialization failure without depending on a real out-of-memory event?
- What evidence distinguishes correct RAII ownership from a long-run leak claim?

After `11_yolov8_trt_cpp`, you should be able to answer:

- How does YOLO preprocessing work?
- How do you allocate and bind TensorRT input/output buffers?
- Where are the main latency costs?
- What parts run on CPU and what parts run on GPU?
- How do you validate TensorRT output against PyTorch output?

After `12_end_to_end_validation_report`, you should be able to answer:

- Can another engineer reproduce the pipeline from a clean environment?
- Which evidence establishes functional correctness, and which claims are still unproven?
- How are TensorRT, CUDA, and OpenCV resources owned across the pipeline?

After `16_cpp_producer_consumer`, `17_dynamic_batching`, and `19_multistream_video_pipeline`, you
should be able to answer:

- Why is a single video-reading loop not enough for industrial camera systems?
- How do `std::mutex` and `std::condition_variable` work together?
- What should happen when the producer is faster than the consumer?
- How do you build a TensorRT engine with dynamic batch profiles?
- How do you calculate offsets for batched `NCHW` input buffers?
- How do you keep stream identity when frames from multiple cameras are batched together?
- How do you choose between fairness and latest-frame freshness?
- What metrics do you report for each stream?

After `25_onnx_graph_surgery_plugin` and `26_custom_tensorrt_plugin`, you should be able to answer:

- What do you do when TensorRT does not support an ONNX operator?
- When should you rewrite PyTorch code instead of writing a plugin?
- What does ONNX GraphSurgeon do?
- What are the core responsibilities of a TensorRT dynamic plugin?
- How is a plugin registered, serialized, deserialized, and called from `enqueue`?
- How do you validate plugin output against a reference implementation?

After `13_nsight_performance_diagnosis` and `20_cuda_preprocess_npp`, you should be able to answer:

- How do you prove GPU starvation from a timeline?
- How do pinned memory and async copies affect overlap?
- When is GPU preprocessing worth the added complexity?
- How do you compare two optimization attempts fairly?

After `15_precision_performance_report` and `22_pipeline_performance_report`, you should be able to answer:

- How were warmup, synchronization, sample count, and percentile latency defined?
- How do you prevent calibration/validation leakage and enforce a predeclared accuracy gate?
- Why can raw tensor drift pass while task-level mAP still regresses, or the reverse?
- What is the difference between inference latency and capture-to-result latency?
- Which bottleneck was proven by evidence, and which optimization did not help?
- How does overload affect queue depth, dropped frames, fairness, and freshness?
- Which failures were injected, which sanitizers were applicable, and what does a 30-minute soak test
  still not prove?

After `24_triton_inference_server` and `29_cpp_shared_library_python_binding`, you should be able to answer:

- How does Triton discover, configure, and version a TensorRT model?
- How do dynamic batching, queue delay, concurrency, and model instances interact?
- Which latency is measured by the client, and which components are measured by the server?
- How do you expose a C++ TensorRT engine to Python safely?

After `27_deepstream_gstreamer_multistream` and `28_jetson_orin_xavier_dla_deployment`, you should be able to answer:

- What is a GStreamer pipeline?
- What does `nvstreammux` do?
- What does zero-copy mean in DeepStream?
- What changes when deploying on Jetson instead of a desktop GPU?
- What constraints determine whether a layer can run on DLA?

After `30_llm_inference_intro`, you should be able to answer:

- What are prefill and decode?
- What is KV cache?
- What are TTFT and time per output token, and why must output length be controlled in a comparison?
- Why is LLM decoding often memory-bandwidth bound?
- How is LLM batching different from YOLO image batching?
- How did input length and batch or concurrency affect measured throughput, latency, and memory use?
- What problems do TensorRT-LLM, OpenVINO GenAI, vLLM, and llama.cpp try to solve?

After `32_final_portfolio_case_study`, you should be able to answer:

- What is the best latency you achieved on the recorded RTX 4090 platform?
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
