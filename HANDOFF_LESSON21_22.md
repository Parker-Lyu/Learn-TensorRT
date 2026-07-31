# Handoff: Finish Lessons 21 and 22

## Objective

Finish `21_integrated_tensorrt_video_pipeline` and `22_pipeline_performance_report` against the
contracts in `docs/learning_roadmap.md`. Do not treat the current generated report's `PASS` as proof
that the roadmap is complete: its gates are narrower than the roadmap acceptance criteria.

## Environment

- Repository: `/home/parker/Code/Learn-TensorRT`
- Running development container: `learn-tensorrt`
- Container workdir: `/workspace/Learn-TensorRT`
- Image: `learn-tensorrt:25.11`, derived from `nvcr.io/nvidia/pytorch:25.11-py3`
- GPU: NVIDIA GeForce RTX 4090, compute capability 8.9
- TensorRT: 10.14.1.48
- CUDA Toolkit/runtime: 13.0
- Run builds and GPU tests inside `learn-tensorrt`.
- Commit every completed code-change phase. Run `git diff --check` before every final handoff.

Useful shell prefix:

```bash
docker exec learn-tensorrt bash -lc 'cd /workspace/Learn-TensorRT && <command>'
```

## Current Repository State

At handoff creation, the worktree is clean except for this handoff commit. Relevant recent commits:

```text
c946d31 docs: reflect reproducible pipeline reliability gates
a0b3d5c fix: support reproducible TSAN evidence on restricted kernels
59901da docs: align lesson 21 README with implemented pipeline
9e88533 feat: record TensorRT CUDA and GPU identity
4a81be8 feat: report batch and per-stream distributions
a3e5b2a feat: support video sequences and latest-first scheduling
a715547 test: verify overload and batch consistency
c7e19b9 fix: report submitted frame accounting
d5e37bf test: add integrated failure injection and cleanup
da4ce52 feat: connect bounded capture queues to GPU scheduler
efa0f00 test: cover bounded queues and multi-source GPU identity
572f2dc feat: add integrated NPP TensorRT YOLO pipeline
```

Lessons 21 onward have already been renumbered. Do not repeat the renumbering migration.

## What Currently Works

### Lesson 21

- Real TensorRT `enqueueV3()` with dynamic batch sizes 1, 2, and 4.
- One shared engine and slot-local execution contexts, streams, events, input/output buffers.
- NPP letterbox resize followed by CUDA BGR-to-RGB, normalization, and HWC-to-CHW.
- Two batches submitted on separate slots before collection.
- YOLO output slicing, decode, NMS, coordinate restoration, JSONL detections, annotated image.
- Per-source bounded queues with block and drop-oldest policies.
- Round-robin and latest-first scheduling, timeout micro-batching.
- Image and video-file inputs, multi-source identity, overload accounting.
- Submit and postprocessing fault hooks.
- Environment, batch distribution, aggregate timing, and per-stream processed counts.

Current test command:

```bash
docker exec learn-tensorrt bash -lc '
  cd /workspace/Learn-TensorRT &&
  cmake -S 21_integrated_tensorrt_video_pipeline \
        -B 21_integrated_tensorrt_video_pipeline/build -G Ninja &&
  cmake --build 21_integrated_tensorrt_video_pipeline/build -j4 &&
  ctest --test-dir 21_integrated_tensorrt_video_pipeline/build --output-on-failure
'
```

Last result: 15/15 tests passed.

### Lesson 22

- Collector runs the real lesson 21 executable as its primary integrated measurement.
- 100 restart cycles, fault matrix, short/formal soak loop, compute-sanitizer and TSAN fields.
- Report generator and two Python tests.
- A formal invocation was actually run:

```text
soak requested: 30.0 minutes
soak cycles: 2407
soak failures: 0
restart cycles: 100
restart failures: 0
compute-sanitizer return code: 0
TSAN return code: 0
```

Generated ignored files:

```text
22_pipeline_performance_report/outputs/evidence.json
22_pipeline_performance_report/outputs/tsan.json
reports/22_pipeline_performance.md
```

The generated report currently says `PASS`, but that status is too broad. Fix the gate definition
before treating it as formal completion.

## Known Gaps and Design Problems

### 1. Video input is not a true streaming source

`pipeline_app.cpp` decodes a video into a `std::vector<cv::Mat>` before capture workers start.
Consequences:

- video decode memory is outside bounded queues;
- a long video can consume memory proportional to the requested frame count;
- capture timestamps do not represent actual `VideoCapture::read()` time;
- file decode is excluded from capture-to-result latency.

Required fix:

- introduce a narrow `FrameSource` interface;
- implement repeatable image, image-sequence, and streaming `VideoCapture` sources;
- each capture worker must call `read()` and immediately timestamp/enqueue a frame;
- queue capacity must bound decoded frames;
- normal EOS closes only that source queue and drains accepted work;
- source failure must follow explicit isolate-or-stop policy.

### 2. Slot-local preprocessing storage is not fully reusable

`tensorrt_backend.cu` allocates/frees source and resized temporary buffers during submissions.
This violates the roadmap's reusable slot-local intermediate-buffer boundary and contaminates
steady-state timing.

Required fix:

- determine/document a maximum accepted source size or grow slot buffers explicitly;
- each slot owns reusable device source and resized/letterbox buffers;
- capacity growth occurs before submission or is reported separately from steady state;
- reject insufficient capacity before unsafe execution;
- do not call `cudaMalloc/cudaFree` per steady-state frame/batch.

### 3. Missing pinned host staging

The backend copies from ordinary OpenCV memory and downloads into `std::vector<float>`. Pageable
memory means `cudaMemcpyAsync` is not a reliable asynchronous ownership boundary.

Required fix:

- each slot owns reusable pinned host input staging for every batch item;
- each slot owns reusable pinned host output staging;
- copy `cv::Mat` into pinned staging while measuring host staging time;
- issue H2D/D2H from pinned storage on the slot stream;
- copy completed output into an owned result only after the completion event;
- document the unavoidable CPU copy from OpenCV decode memory.

### 4. GPU slot state machine is duplicated

The CPU-tested `SlotPool` and the GPU app's `free`/`pending` deques are separate state machines.
The CPU test therefore does not prove the real GPU path uses the validated transitions.

Required fix:

- make one slot-pool owner control `Free -> Reserved -> Submitted -> Completing -> Free`;
- backend submission and collector APIs must transition through that owner;
- only the collector can return a submitted slot to free;
- failure states must distinguish reserved-but-not-submitted from GPU-submitted work;
- reverse-completion CPU tests and real GPU tests must use the same state-transition component.

### 5. Timing schema is incomplete

Current preprocessing time combines multiple stages, and metrics mostly contain accumulated totals.

Required per-batch samples and summaries:

- queue waiting time;
- batch-fill waiting time;
- host staging time;
- H2D time;
- NPP resize/fused preprocessing time;
- TensorRT GPU execution time;
- D2H time;
- CPU decode/NMS/postprocessing time;
- capture-to-result latency;
- batch-size distribution;
- per-stream and aggregate throughput plus P50/P90/P99.

Use host `steady_clock` only for host/end-to-end durations and CUDA events for GPU stages. Document
clock domains; do not subtract timestamps from different domains.

### 6. Failure coverage is incomplete

Current hooks cover submit boundary and CPU postprocessing. Add distinct, testable failures for:

- source/capture read;
- pinned host allocation/staging;
- NPP resize;
- CUDA kernel launch or asynchronous completion;
- TensorRT runtime shape/profile rejection;
- `setTensorAddress`/capacity validation;
- `enqueueV3()`;
- output transfer;
- CPU postprocessing.

Abort semantics must be explicit:

1. stop new source reads and submissions;
2. discard and account for unsubmitted queued/batched frames;
3. quiesce, not “cancel”, already submitted CUDA work;
4. observe asynchronous errors from event/stream completion;
5. join all workers before TensorRT/CUDA resource destruction;
6. return nonzero with the first causal error retained.

### 7. `pipeline_app.cpp` needs production-style refactoring

It is currently effectively one very long line with CLI parsing, source loading, scheduling,
collection, postprocessing, metrics, JSON, and annotation in one function. This violates the
repository's industrial teaching-code expectations.

Refactor into small files/classes, for example:

```text
include/config.hpp                 CLI/config validation
include/frame_source.hpp           image/video source interface
include/integrated_pipeline.hpp    owner and run result
include/metrics.hpp                samples, counters, percentiles, JSON schema
include/result_writer.hpp          detections and annotation output
src/config.cpp
src/frame_source.cpp
src/integrated_pipeline.cpp
src/metrics.cpp
src/result_writer.cpp
src/main.cpp
```

Avoid adding framework layers that do not serve these concrete boundaries.

### 8. Lesson 22 formal soak is not a long-lived-process soak

The 30-minute collector ran 2407 short processes. This is useful lifecycle stress, but it is not a
30-minute continuously running inference service. It cannot prove absence of long-lived allocator,
buffer, queue, thread, or metrics growth.

Required fix:

- add a duration or repeat mode to lesson 21 that keeps one pipeline process alive for 30 minutes;
- continuously recycle slots and sources in that process;
- periodically emit/snapshot queue depth, RSS, device memory, processed/dropped counts, and errors;
- keep the existing 100-process restart campaign as a separate gate;
- make `soak_30_minutes` require the long-lived mode, not a loop of short processes.

### 9. Memory gate does not establish a trend

The last evidence sampled one short integrated process. Host RSS ended at its peak because sampling
stopped before process exit; that alone is not a leak, but it also does not prove stability.

Required fix:

- save time-series samples during the long-lived process;
- define warm-up exclusion;
- compare bounded windows or calculate a documented trend slope;
- record queue depths and work count with memory samples;
- distinguish allocator high-water retention from monotonic growth;
- device-memory sampling must target the lesson process, not indiscriminately sum all compute apps;
- gate thresholds must be documented and tested with fixtures.

### 10. Sanitizers do not directly cover all lesson 21 paths

- Compute-sanitizer currently focuses primarily on the lesson 20 CUDA test.
- TSAN evidence covers the lesson 16 producer-consumer test, not the lesson 21 scheduler/owner.

Required fix:

- run compute-sanitizer memcheck directly on lesson 21 batch, two-slot, overload, and failure-cleanup
  smoke cases;
- run applicable `initcheck`, `synccheck`, and racecheck only where meaningful and document limits;
- add `ENABLE_TSAN` support for lesson 21 CPU-only core/scheduler tests;
- execute lesson 21 queue, scheduler, shutdown, and failure tests under TSAN;
- keep TSAN separate from CUDA/TensorRT processes.

On this kernel the first TSAN run reported `unexpected memory mapping`. A verified workaround was:

```bash
docker run --rm --security-opt seccomp=unconfined \
  -v "$PWD:/workspace/Learn-TensorRT" -w /workspace/Learn-TensorRT \
  learn-tensorrt:25.11 \
  bash -lc 'setarch x86_64 -R 16_cpp_producer_consumer/build-tsan/producer_consumer_tests'
```

Do not interpret a TSAN startup failure as either a code failure or a passing sanitizer run.

### 11. Multi-stream report evidence is partly simulated

The report's fairness section still consumes lesson 19 deterministic-worker measurements.

Required fix:

- make the primary multi-stream matrix run lesson 21 real TensorRT with at least two sources;
- collect per-stream latency, throughput, queue peak, accepted, dropped, and freshness;
- compare round-robin and latest-first using the same real engine/input/hardware;
- retain lesson 19 results only as supporting CPU scheduling tests, clearly labelled simulated.

### 12. Current report gates are too permissive

`generate_report.py::evaluate()` checks only a small set of booleans. It currently permits a global
`PASS` without proving all roadmap criteria.

Add explicit gates for at least:

- real integrated batch 1/2/4;
- two overlapping slots;
- batch versus single-image result tolerance;
- CPU versus CUDA preprocessing tolerance;
- real multi-stream identity and per-stream metrics;
- block and drop-oldest accounting invariants;
- latest-first freshness accounting;
- long-lived 30-minute soak;
- 100 restarts;
- host/device memory trend;
- lesson 21 compute-sanitizer;
- lesson 21 CPU TSAN;
- complete integrated fault matrix;
- environment/schema completeness.

A missing field or unavailable tool must produce `INCOMPLETE`, not an exception and not `PASS`.
Add report tests for passing, incomplete, malformed, and failed evidence fixtures.

## Recommended Work Plan

### Phase 1: Correct the evidence contract first

1. Translate every lesson 21/22 roadmap acceptance criterion into an evidence field and report gate.
2. Change the current global report from overly broad `PASS` to `INCOMPLETE` until those fields exist.
3. Add versioned evidence schemas and Python tests for complete/incomplete/malformed inputs.
4. Commit this phase separately.

### Phase 2: Refactor lesson 21 without changing validated output

1. Split `pipeline_app.cpp` into config/source/pipeline/metrics/result modules.
2. Preserve current batch-1/batch-4 detections as regression evidence.
3. Integrate the tested slot state machine with real GPU slots.
4. Run all existing CPU/GPU tests after each substep.
5. Commit the refactor separately from behavior changes.

### Phase 3: Implement real streaming and reusable memory

1. Implement `FrameSource` and streaming `VideoCapture`.
2. Add pinned staging and slot-local reusable device intermediates.
3. Add capacity validation and remove steady-state allocation.
4. Split timing events and host timing samples.
5. Add focused preprocessing and resource-reuse tests.
6. Run compute-sanitizer on the integrated executable.
7. Commit.

### Phase 4: Complete lifecycle, failure, and metrics behavior

1. Add explicit drain/abort owner semantics and integrated slot-pool transitions.
2. Add all missing fault injection points and cleanup tests.
3. Add per-stream latency/throughput/freshness and complete terminal accounting.
4. Add lesson 21 TSAN configuration and run CPU paths under TSAN.
5. Commit.

### Phase 5: Rebuild lesson 22 formal evidence

1. Add long-lived duration mode to lesson 21.
2. Collect process-specific memory time series and trend evidence.
3. Run real TensorRT single-/multi-stream policy matrix.
4. Run 100 restarts separately.
5. Run direct lesson 21 compute-sanitizer and lesson 21 CPU TSAN.
6. Run the formal 30-minute continuous soak only after short fixtures pass.
7. Generate the final report and verify every gate against saved evidence.
8. Keep generated `reports/` and local `outputs/` ignored as required.
9. Commit code, tests, README, roadmap/coverage wording changes—not generated local evidence.

### Phase 6: Completion audit

Before saying “complete”:

```bash
# Lesson 21 build/tests
cmake -S 21_integrated_tensorrt_video_pipeline \
      -B 21_integrated_tensorrt_video_pipeline/build -G Ninja
cmake --build 21_integrated_tensorrt_video_pipeline/build --parallel
ctest --test-dir 21_integrated_tensorrt_video_pipeline/build --output-on-failure

# Lesson 22 tests
python3 -m unittest discover -s 22_pipeline_performance_report/tests -v

# Formal evidence (after short validation)
python3 22_pipeline_performance_report/collect_pipeline_evidence.py \
  --soak-minutes 30 --restart-cycles 100
python3 22_pipeline_performance_report/generate_report.py

git diff --check
git status --short
```

Audit each roadmap acceptance criterion against direct source/test/evidence. A green aggregate test
count is not sufficient. State exact hardware/container limitations. Commit all code changes.

## Important Interpretation

The existing code is valuable and should be refactored rather than discarded. However, do not claim
that lessons 21 and 22 are completely finished until the long-lived soak, direct lesson 21
sanitizers, reusable pinned/slot-local memory, true streaming source, complete failure matrix, and
strict report gates are implemented and verified.
