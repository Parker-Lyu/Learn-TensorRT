# ModelOpt ONNX And TensorRT FP16 Workflow Handoff

## Decision

The development image will provide the ModelOpt ONNX toolchain globally. Lesson 06 will own both
TensorRT FP16 build styles:

1. **Legacy compatibility path:** build an ordinary FP32 ONNX graph with `trtexec --fp16`. This
   demonstrates the weakly typed builder workflow still found in older production systems.
2. **Modern path:** use ModelOpt AutoCast to produce an explicit mixed FP16/FP32 ONNX graph, validate
   that graph, and build it with `trtexec --stronglyTyped`.

Lesson 05 remains the canonical PyTorch-to-FP32-ONNX export and validation boundary. Lesson 17
consumes lesson 06's validated dynamic AutoCast ONNX graph and focuses on its own batch profile,
runtime shapes, buffer offsets, and throughput/latency trade-off.

## Initial Environment Findings (Before The Curated Stack)

The following findings describe the starting image state that motivated the curated stack now
recorded in the implementation status below. The pinned upstream image supplies
`nvidia-modelopt==0.37.0`, but did not provide a working ModelOpt ONNX environment as shipped:

- `onnx-graphsurgeon` is absent.
- Installing GraphSurgeon alone against the current `onnx==1.21.0` fails because that combination
  expects an ONNX helper removed from 1.21.
- ModelOpt 0.37 declares `onnx~=1.19.0` and selects `onnxruntime-gpu~=1.22.0` on Linux for its ONNX
  extra. The course intentionally installs the CPU `onnxruntime==1.22.0` distribution instead.
  Lesson 05 and ModelOpt AutoCast use the CPU provider as a reproducible ONNX numerical-validation
  reference; TensorRT remains the course's GPU deployment runtime. `onnxruntime-gpu` is not an
  invalid package, but it is an optional alternative for a separate ONNX Runtime CUDA-EP workflow,
  not a prerequisite for the TensorRT lessons. Keeping the CPU distribution also satisfies tools
  that check Ultralytics' export dependency by the distribution name `onnxruntime`.
- ModelOpt's published extra selects `cupy-cuda12x`, while this course targets CUDA 13.0. The image
  must install the equivalent curated dependency set with `cupy-cuda13x`, rather than silently add
  a CUDA 12 CuPy wheel.
- The upstream `nvidia-resiliency-ext` distribution has a stale `pynvml` package-name requirement.
  The image retains NVIDIA's maintained `nvidia-ml-py` binding instead of installing deprecated
  `pynvml`; `pip check` therefore reports that known upstream metadata issue.

A temporary compatibility test using ONNX 1.19.1 and GraphSurgeon 0.5.8 successfully converted the
dynamic YOLOv8n graph, kept FP32 input/output types, converted 165 of 323 nodes to FP16, passed the
ONNX checker, built a TensorRT 10.14 strongly typed engine, and ran batch sizes 1, 2, and 4. That
test established feasibility, not a committed accuracy result.

## Planned Changes

### Development image

- Align the global ONNX toolchain with ModelOpt 0.37.
- Install the ModelOpt ONNX dependencies globally, using the CUDA 13 CuPy distribution.
- Keep the NVIDIA-provided PyTorch, CUDA, TensorRT, and ModelOpt builds unchanged.
- Add image-build assertions for ModelOpt AutoCast imports, ONNX Runtime providers, and the pinned
  dependency versions.
- Extend lesson 00 environment verification and setup documentation so a partial ModelOpt install
  fails visibly.

### Lesson 06

- Add an AutoCast preparation/validation tool that:
  - consumes lesson 05's saved validated FP32 input;
  - prepares representative static and dynamic inputs;
  - runs ModelOpt AutoCast for static and dynamic ONNX models;
  - preserves FP32 model I/O;
  - keeps the YOLO detection head in FP32;
  - checks the converted model structurally;
  - runs ONNX Runtime on the FP32 and AutoCast graphs with the same input;
  - writes reproducible JSON validation evidence and fails on declared tolerances.
- Keep the existing weakly typed `--fp16` builds and label them explicitly as legacy/deprecated.
- Add strongly typed static and dynamic FP16 builds from the validated AutoCast ONNX graphs.
- Record build mode, source ONNX, precision contract, deprecation status, and validation artifact in
  the build manifest and benchmark summary.
- Add focused CPU tests for AutoCast input preparation, output comparison, build planning, and
  `trtexec` command generation.

### Lesson 17 and downstream consumers

- Change lesson 17's engine builder to consume lesson 06's validated dynamic AutoCast ONNX and use
  `--stronglyTyped`; do not make lesson 17 run AutoCast itself.
- Document the exact lesson 06 preparation command as a cross-lesson prerequisite.
- Keep the lesson 17-specific batch profile at min/opt/max batch 1/2/4.
- Update lesson 29 and lesson 32 commands and artifact names where they depend on lesson 17 or the
  static delivery engine.
- Keep generated ONNX graphs, engines, timing caches, logs, reports, and benchmarks ignored.

### Course contracts

- Update `docs/learning_roadmap.md` so lesson 06 explicitly teaches both legacy weakly typed and
  modern strongly typed FP16 paths without changing lesson 05's acceptance boundary.
- Update learner-facing READMEs to state that weakly typed mode is retained for production legacy
  literacy, not presented as the recommended new implementation.

## Verification Plan

1. Rebuild `learn-tensorrt:25.11` from `docker/Dockerfile.dev` and recreate the persistent
   `learn-tensorrt` container.
2. Run `00_environment_check/check_env.sh` and confirm the ModelOpt ONNX import/version checks pass.
3. Re-run lesson 05 export and FP32 validation with the aligned global ONNX Runtime stack.
4. Run lesson 06 AutoCast generation and ONNX Runtime validation for static and dynamic graphs.
5. Dry-run all lesson 06 build modes and run focused unit tests.
6. Smoke-build at least one legacy and one strongly typed engine; build the modern dynamic engine.
7. Build and run lesson 17 for batch sizes 1, 2, and 4 with the strongly typed engine.
8. Run affected lesson 32 tests, shell syntax checks, `git diff --check`, and document any expensive
   benchmark or delivery-image verification not performed.

## Risks And Boundaries

- Changing global ONNX and ONNX Runtime versions affects lessons 05, 06, 07, and 14; their focused
  checks must be rerun before the environment change is accepted.
- A successful AutoCast or engine build does not prove numerical correctness. The lesson 06 JSON
  gate must compare raw outputs on identical inputs.
- One-image raw-output alignment is a conversion gate, not dataset-level detection-quality proof.
- The detection-head exclusion is model-graph-specific and must fail visibly if future export node
  names no longer match the expected `/model.22/` prefix.
- Serialized engines and benchmark numbers remain environment-specific and must not be committed.

## Implementation Status (2026-08-07)

The approved design is now implemented. The development image has the globally pinned ModelOpt ONNX
stack; lesson 06 generates and validates static/dynamic AutoCast graphs using semantic YOLO output
tolerances (5% relative/absolute for pixel-scale box coordinates and 1%/0.02 for scores). Lesson 06
build planning emits both deprecated weakly typed and recommended strongly typed commands. Lesson 17
and the lesson 32 delivery helper consume the validated graphs with `--stronglyTyped`.

Verified in the persistent container: lesson 05 export/runtime validation, static and dynamic lesson
06 conversion, strongly typed static and dynamic TensorRT builds, lesson 17 dynamic-profile engine
build (TensorRT reported `PASSED`), Python tests, shell syntax, and `git diff --check`. Full-duration
benchmark matrices and lesson 17 C++ runtime batch measurements remain environment-specific and were
not rerun in this change.
