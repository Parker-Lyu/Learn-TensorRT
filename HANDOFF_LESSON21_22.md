# Handoff: Complete Lessons 21 and 22

## Objective

Complete Lessons 21 and 22 against `docs/learning_roadmap.md` without mixing their acceptance
boundaries:

- Lesson 21 delivers a correct, bounded, observable integrated TensorRT pipeline.
- Lesson 22 produces environment-specific performance and reliability evidence from Lesson 21.

Lesson 21 does not claim soak, restart, sanitizer, Nsight, or production-performance evidence.
Those claims belong to Lesson 22. Missing formal evidence must make the Lesson 22 report
`INCOMPLETE`; it must never be inferred from a green unit-test count.

## Environment

- Development image: `nvcr.io/nvidia/pytorch:25.11-py3`
- Local image/container: `learn-tensorrt:25.11` / `learn-tensorrt`
- TensorRT: 10.14.1.48
- CUDA Toolkit/runtime: 13.0
- Recorded GPU: NVIDIA GeForce RTX 4090, compute capability 8.9

Run CUDA and TensorRT checks inside the pinned development container:

```bash
docker exec learn-tensorrt bash -lc 'cd /workspace/Learn-TensorRT && <command>'
```

Generated engines, reports, sanitizer logs, benchmark captures, and local output files remain
ignored.

## Current State

The repository already has real `enqueueV3()`, dynamic batches 1--4, two GPU slots, NPP/CUDA
preprocessing, YOLO decode/NMS, identity-bearing output, bounded queue policies, multi-source
scheduling, overload tests, and basic fault hooks. Preserve this behavior while fixing the
ownership and evidence gaps below.

## Lesson 21 Required Work

### 1. Incremental frame sources

Introduce a narrow `FrameSource` abstraction and implementations for repeatable images,
image sequences, synthetic frames, and incrementally decoded video files. Do not decode the whole
requested video into `std::vector<cv::Mat>` before capture starts. A capture worker reads one frame,
timestamps it with `steady_clock`, and immediately offers it to its bounded source queue.

Use stop-all semantics for the first implementation: retain the first source error, stop new reads
and submissions, account for unsubmitted work, quiesce submitted GPU work, join workers, then exit
nonzero. Normal EOS closes that source queue and drains accepted work.

### 2. Reusable per-slot resources

Each slot owns its execution context, stream, events, device input/output, device source/letterbox
intermediates, pinned input staging, and pinned output staging. No steady-state submission may call
`cudaMalloc`, `cudaFree`, `cudaMallocHost`, or `cudaFreeHost`. Capacity growth occurs before a
submission and is reported separately. Empty, oversized, profile-invalid, and insufficient-capacity
inputs are rejected before `enqueueV3()`.

The unavoidable copy from decoded OpenCV memory into pinned staging is measured as host staging.

### 3. One slot owner

The CPU-tested state owner must control the real GPU path:

```text
Free -> Reserved -> Submitted -> Completing -> Free
                         |
                         -> Failed
```

Only the collector releases submitted slots. Tests cover invalid transitions, reverse completion,
reserved failures, submitted failures, drain, abort, and double release. GPU tests cover batches
1/2/4, two submitted slots, invalid profile/capacity, and cleanup with pending work.

### 4. Timing and accounting

Use host `steady_clock` for queue wait, batch-fill wait, host staging, CPU postprocessing, and
capture-to-result latency. Use CUDA events for H2D, CUDA/NPP preprocessing, TensorRT, and D2H. Never
subtract timestamps from different clock domains.

Save per-batch samples and aggregate P50/P90/P99, batch distribution, queue peaks, environment
identity, and aggregate/per-stream throughput and latency. Every captured frame reaches exactly one
terminal category:

```text
captured = completed + evicted + failed + aborted
```

`submitted` counts only frames passed to the backend. Normal EOS drains. Abort stops new work,
accounts for queued/batched frames, quiesces submitted CUDA work, retains the first causal error,
and returns nonzero.

### 5. Application structure

Replace the monolithic application with focused modules:

```text
include/config.hpp              src/config.cpp
include/frame_source.hpp        src/frame_source.cpp
include/integrated_pipeline.hpp src/integrated_pipeline.cpp
include/metrics.hpp             src/metrics.cpp
include/result_writer.hpp       src/result_writer.cpp
src/main.cpp
```

Do not introduce unrelated framework layers.

## Lesson 22 Required Work

### 1. Versioned evidence and tri-state gates

Schema version 3 separates platform, load matrix, policies, references, faults, restarts,
long-lived soak, memory series, and sanitizers. Every required gate is `PASS`, `FAIL`, or
`INCOMPLETE` (`NOT_APPLICABLE` is allowed only for documented optional tools).

- Overall `PASS`: all required gates pass.
- Overall `FAIL`: at least one executed required gate fails.
- Overall `INCOMPLETE`: no required gate fails, but evidence is absent/unavailable/malformed.

Missing fields must not raise `KeyError` and must not pass.

Required gates cover real batches 1/2/4, two slots, reference tolerances, real multi-stream identity
and metrics, block/drop-oldest/latest-first accounting, bounded queues/in-flight work, integrated
faults, 100 restarts, one-process 30-minute soak, host/device memory trend, direct Lesson 21
compute-sanitizer, Lesson 21 CPU TSAN, and environment/schema completeness.

Lesson 19 may appear only as labelled supporting CPU scheduling evidence.

### 2. Long-lived soak and memory trend

Add a Lesson 21 duration/repeat mode that keeps one process alive. It periodically records queue
depth, slot states, RSS, process-specific device memory, work counters, and errors. The 30-minute
gate uses this mode; the 100-process restart campaign remains separate.

Exclude a documented warm-up and compare bounded windows rather than start/peak/end alone. Sample
the lesson PID, not all compute applications. Store thresholds in evidence and test them with
fixtures. Default formal thresholds are 60 seconds warm-up, 120-second windows, at most 5% RSS and
device-memory window growth, and queue/in-flight peaks no greater than configured capacities.

### 3. Sanitizers

Run compute-sanitizer memcheck directly on Lesson 21 batch/two-slot and selected integrated
overload/failure cleanup cases. Build CPU-only Lesson 21 core, scheduler, ownership, shutdown, and
failure tests with TSAN; do not put CUDA/TensorRT processes under TSAN. A TSAN startup failure is
`INCOMPLETE`, not a pass or a code failure.

## Execution Phases

1. Define schema v3, tri-state evaluation, and passing/incomplete/malformed/failed fixtures. The
   current broad report becomes `INCOMPLETE`.
2. Refactor Lesson 21 without changing validated detections; integrate `FrameSource` and the real
   slot owner.
3. Add pinned staging, reusable device intermediates, capacity checks, and split timings.
4. Complete drain/abort, terminal accounting, reproducible fault hooks, and CPU TSAN targets.
5. Collect the real single/multi-stream policy matrix, separate restart and long-lived soak, sample
   process memory, and run direct sanitizers.
6. Audit every roadmap criterion against source, a focused test, and saved evidence.

Commit each completed implementation phase separately.

## Completion Commands

```bash
cmake -S 21_integrated_tensorrt_video_pipeline \
      -B 21_integrated_tensorrt_video_pipeline/build -G Ninja
cmake --build 21_integrated_tensorrt_video_pipeline/build --parallel
ctest --test-dir 21_integrated_tensorrt_video_pipeline/build --output-on-failure

python3 -m unittest discover -s 22_pipeline_performance_report/tests -v

# Run only after short evidence fixtures pass.
python3 22_pipeline_performance_report/collect_pipeline_evidence.py \
  --soak-minutes 30 --restart-cycles 100
python3 22_pipeline_performance_report/generate_report.py

git diff --check
git status --short
```

Do not claim an unexecuted GPU, sanitizer, soak, restart, or hardware-specific gate passed. Record
the exact limitation as `INCOMPLETE`.
