# 21 - Integrated TensorRT Video Pipeline

## Purpose

Integrate the bounded multi-stream scheduling semantics from lessons 16–19 with lesson 20 CUDA/NPP
preprocessing and real TensorRT dynamic execution. This lesson establishes correctness, ownership,
and measurement boundaries; lesson 22 owns formal soak and performance conclusions.

## Prerequisites

Run all commands below from the repository root inside the persistent `learn-tensorrt` container
configured in [lesson 00](../00_environment_check/README.md). Follow
[lesson 17's prerequisites](../17_dynamic_batching/README.md#prerequisites), then build its dynamic
batch 1–4 engine:

```bash
./17_dynamic_batching/build_dynamic_engine.sh
```

`assets/img.jpeg` is the default controlled image. The CMake targets reuse lesson 11's committed
`postprocess.cpp` and `geometry.cpp`; no lesson 11 build products are required.

## Deliverables

- `integrated_pipeline_core`: CPU-testable immutable metadata, slot lifecycle, accounting, and dispatch.
- `integrated_tensorrt_video_pipeline_gpu`: runnable bounded multi-source NPP/TensorRT/YOLO pipeline.
- `integrated_pipeline_reporting`: CPU-testable result handling and metrics.
- CPU-focused concurrency tests and separately documented GPU integration, consistency, overload, and failure tests.

## Design

The shared engine outlives every slot. Each slot exclusively owns its context, stream, lifecycle
completion event, timing events, reusable device intermediates, and pinned host staging. The tested
slot owner also controls the real backend: the submitter performs `Free -> Reserved -> Submitted`;
the collector performs `Submitted -> Completing -> Free`. Completion order never defines identity.
The backend checks its model contract before using tensor buffers: exactly one input/output,
FP32 linear device I/O, NCHW input `[N,3,640,640]`, and YOLO output `[N,4+classes,candidates]`.
An engine with FP16 internal computation may still expose FP32 I/O.
Normal EOS drains. Abort rejects new work, discards unsubmitted work, and quiesces submitted CUDA
work before resource destruction; already submitted CUDA work is not described as cancellable.

Image, image-sequence, synthetic, and video-file sources implement one incremental read interface.
Video decode therefore occurs inside the capture worker and decoded frames are bounded by the
per-source queues; a whole video is never preloaded. Counters distinguish captured, admitted, queue eviction, submission,
completion, in-flight failure, and abort discard. CUDA events measure GPU stages; the monotonic host
clock measures waiting and capture-to-dispatch latency. See the English
[code walkthrough](CODE_WALKTHROUGH.md) or [中文代码导读](CODE_WALKTHROUGH_ZH.md) for the reading order.

## Setup

The optional CPU ThreadSanitizer run uses Clang 18.1.3 and its matching compiler runtime. New
images built from `docker/Dockerfile.dev` include both. For an existing persistent container,
install the pinned tools with this container-management command **on the host**:

```bash
docker exec --user root learn-tensorrt bash -lc \
  'apt-get update && apt-get install -y --no-install-recommends \
    clang-18=1:18.1.3-1ubuntu1 libclang-rt-18-dev=1:18.1.3-1ubuntu1'
```

This adds a CPU test toolchain; ordinary lessons continue to use their existing compiler and
CUDA/TensorRT/PyTorch stack. Apt may update distribution compiler/runtime dependencies; the local
verification used glibc `2.39-0ubuntu8.9`, GCC `13.3.0-6ubuntu2~24.04.1`, and Clang
`1:18.1.3-1ubuntu1`. The course environment check and normal GPU suite were rerun afterward.

## Build

```bash
cmake -S 21_integrated_tensorrt_video_pipeline -B 21_integrated_tensorrt_video_pipeline/build
cmake --build 21_integrated_tensorrt_video_pipeline/build --parallel
```

## Run

Run the CPU ownership and reverse-completion smoke test:

```bash
./21_integrated_tensorrt_video_pipeline/build/integrated_tensorrt_video_pipeline
```

Example output (local run):

```text
wrote 2 identity records to 21_integrated_tensorrt_video_pipeline/output/detections.jsonl
This CPU run is not TensorRT performance evidence.
```

This smoke path is explicitly not TensorRT or performance evidence. Formal GPU acceptance requires
the dynamic engine and the pinned GPU container.

Run the real GPU-integrated path with NPP letterbox, TensorRT contexts/streams, dynamic batches, and YOLO decode/NMS:

```bash
./21_integrated_tensorrt_video_pipeline/build/integrated_tensorrt_video_pipeline_gpu \
  17_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine assets/img.jpeg 16 4 2
```

Example output (local container run):

```text
backend=tensorrt captured=16 processed=16 dropped=0 queue_peak=4 fps=142.364
```

This short correctness run includes startup allocation and the first annotation; its FPS is not a
steady-state performance baseline.

The positional arguments after the image are frame count, maximum batch size, and in-flight slot count.
Additional positional arguments select output directory, overload policy, queue capacity, capture
interval, and scheduling policy. Run `synthetic` for a generated 640x480 source, or use
`sequence:path1|path2` for a repeatable image sequence.


## Outputs

`output/detections.jsonl`, `output/metrics.json`, `output/batch_timing_samples.jsonl`,
`output/metrics_snapshots.jsonl`, and `output/annotated_0.jpg` are generated by the real GPU path and remain ignored. The metrics summary
keeps bounded latency reservoirs and at most 256 representative batch records in memory; exact
per-batch timing evidence is streamed to JSONL so a long-running process does not grow with its work
count. Structured records and annotated samples belong under ignored `output/`. Source, tests, and this
README are committed; engines, measurements, and generated images are not.

The metrics schema is version 3. Submission is counted immediately after a successful backend
submit; completion means the entire batch passed postprocessing and result handling. On an abort,
already-submitted batches that are quiesced without successful dispatch count as `failed_in_flight`,
even when their GPU work finished successfully. Frames removed during a failed batch assembly or
failed submit count as `discarded_on_abort`. Terminal metrics validate all three equations:

```text
captured = admitted + rejected_on_admission
admitted = evicted + submitted + discarded_on_abort
submitted = completed + failed_in_flight
```

For lesson 22 compatibility, `failed` aliases `failed_in_flight`, and `aborted` combines
`discarded_on_abort` with `rejected_on_admission`. `normal_drain` distinguishes EOS drain from a
duration stop or failure abort. Runtime failures attempt to write `metrics.json` before rethrowing
the original error; initialization failures before the pipeline starts do not produce a summary.
A partially written batch can leave detection records even though that batch did not complete.

Timing boundaries:

| Field | Measured interval |
| --- | --- |
| `queue_wait_ms` | Batch mean from successful source read to submission preparation; includes admission backpressure and batch fill |
| `batch_fill_wait_ms` | Host time inside `next_batch`; overlaps `queue_wait_ms` and must not be added to it |
| `cpu_postprocess_ms` | Output slicing, metadata preparation, YOLO decode/NMS and coordinate restoration |
| `result_output_ms` | Annotation, JPEG writing and buffered JSONL serialization, excluding filesystem durability guarantees |
| Per-frame latency | Successful source read to result handling, including output work; excludes video decode |

The reported percentile reservoirs retain recent observations rather than the full run. Raw
per-batch JSONL retains the timing samples needed for later analysis.

## Tests

```bash
ctest --test-dir 21_integrated_tensorrt_video_pipeline/build --output-on-failure
```

Example output (partial):

```text
100% tests passed, 0 tests failed out of 24
```

CPU tests cover queue closure/wakeup, scheduler drain/fairness/freshness, immutable pixels, partial
batch failure, slot ownership, accounting, metadata, reverse completion, output validation, and
separate postprocessing/output timing. GPU tests cover NPP/TensorRT batches, independent slots, identity, overload, failure cleanup, and batch consistency; they require the pinned container, lesson 17 engine, and an NVIDIA GPU.

For a build that does not configure CUDA or TensorRT targets:

```bash
cmake -S 21_integrated_tensorrt_video_pipeline \
  -B 21_integrated_tensorrt_video_pipeline/build-cpu -DLESSON21_ENABLE_CUDA=OFF
cmake --build 21_integrated_tensorrt_video_pipeline/build-cpu --parallel
ctest --test-dir 21_integrated_tensorrt_video_pipeline/build-cpu --output-on-failure
```

Example output (partial):

```text
100% tests passed, 0 tests failed out of 3
```

GPU TensorRT smoke (requires the lesson 17 dynamic-profile engine):

```bash
./21_integrated_tensorrt_video_pipeline/build/integrated_tensorrt_gpu_smoke \
  17_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine assets/img.jpeg 1
# Repeat with 2 and 4 to exercise the same optimization profile.
```

The smoke executable reports the real TensorRT output element count and checksum. It is a
backend-level correctness smoke test, not a latency benchmark.

Pass comma-separated repeatable image sources to exercise the same batch/slot path with immutable
per-source identity:

```bash
./21_integrated_tensorrt_video_pipeline/build/integrated_tensorrt_video_pipeline_gpu \
  17_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine \
  assets/img.jpeg,assets/img.jpeg 16 4 2
```

### Failure Experiments

```bash
LESSON21_FAIL_SUBMIT_BATCH=1 ./21_integrated_tensorrt_video_pipeline/build/integrated_tensorrt_video_pipeline_gpu \
  17_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine assets/img.jpeg 8 4 2
LESSON21_FAIL_POSTPROCESS_BATCH=0 ./21_integrated_tensorrt_video_pipeline/build/integrated_tensorrt_video_pipeline_gpu \
  17_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine assets/img.jpeg 8 4 2
```

Each injection is expected to return nonzero. The CTest failure runner also checks the intended
error message and all terminal accounting equations; an unrelated startup error cannot satisfy it.
Both ready-event collection and the blocking fallback execute the same postprocess injection.
Capture queues close and join through their owner, while the TensorRT backend
synchronizes submitted slot streams before destroying CUDA or TensorRT resources.

The optional arguments after `OUTPUT` are overload policy (`block` or `drop-oldest`), per-source
queue capacity, and synthetic capture interval in milliseconds. A controlled overload run is:

```bash
./21_integrated_tensorrt_video_pipeline/build/integrated_tensorrt_video_pipeline_gpu \
  17_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine assets/img.jpeg \
  100 1 1 21_integrated_tensorrt_video_pipeline/output/overload drop-oldest 1 0
```

Its terminal invariant is `captured == processed + dropped`, and queue peak must not exceed the
configured capacity.

A final optional scheduler argument selects `round-robin` (default) or `latest-first`. Latest-first
discards stale queued frames explicitly and includes them in `dropped`; it never hides freshness
losses from the terminal accounting invariant.

Image paths and video-file paths use the same CLI. `VideoCapture::read()` runs incrementally in the
capture worker. The frame timestamp is assigned after a successful decode; capture-to-result
latency therefore starts before bounded-queue admission, while video decode cost remains a
capture-side cost excluded from these latency measurements.

Lesson 22 can keep one Lesson 21 process alive instead of repeatedly launching short processes:

```bash
./21_integrated_tensorrt_video_pipeline/build/integrated_tensorrt_video_pipeline_gpu \
  17_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine assets/img.jpeg \
  16 4 2 21_integrated_tensorrt_video_pipeline/output/soak \
  block 4 0 round-robin --duration-seconds 1800 --repeat-source \
  --metrics-interval-seconds 5 --max-detection-records 1024
```

The duration stops new capture, discards and accounts for queued work, and quiesces submitted work.
Lesson 22 samples the process RSS and process-specific device memory while this command remains alive.

For CPU ThreadSanitizer checks, use a separate Clang 18 build directory inside the container:

```bash
cmake -S 21_integrated_tensorrt_video_pipeline \
      -B 21_integrated_tensorrt_video_pipeline/build-tsan-clang18 \
      -DLESSON21_ENABLE_CUDA=OFF -DLESSON21_ENABLE_TSAN=ON \
      -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_BUILD_TYPE=Debug
cmake --build 21_integrated_tensorrt_video_pipeline/build-tsan-clang18 --parallel
ctest --test-dir 21_integrated_tensorrt_video_pipeline/build-tsan-clang18 --output-on-failure
```

Example output (partial):

```text
100% tests passed, 0 tests failed out of 6
```

The six CTest cases comprise three instrumented CPU tests, two positive controls, and one launcher
suite containing eight Python cases. The positive controls deliberately create a data race and a
lock-order inversion: each must produce its intended TSAN report and exit code 66 to pass. Thus a
missing or disabled detector cannot satisfy the acceptance check. This covers executed CPU paths;
CUDA device accesses and uninstrumented dependency internals need separate tools.

On this host, GCC TSAN failed before `main()` with incompatible randomized addresses (and sometimes
an empty startup SIGSEGV). Clang 18 reports a denied `personality(ADDR_NO_RANDOMIZE)` call explicitly.
The CTest launcher clears HPC-X `LD_PRELOAD` only for the CPU child and retries only exact known
pre-main layout diagnostics, with a 64-attempt/45-second limit. It never retries a race report,
lock report, test failure, timeout, unrecognized crash, or a diagnostic mixed with program output.
Attempts remain visible with `ctest --verbose`. Host ASLR and persistent container settings are not
changed. Startup exhaustion remains a failed test.

The pinned GDAL 3.8.4 library emits a known lock-initialization warning when OpenCV initializes its
codecs. The narrow `deadlock:CPLCreateOrAcquireMutexEx` rule in
[`tests/tsan.supp`](tests/tsan.supp) matches the
[upstream fixed issue](https://github.com/OSGeo/gdal/issues/1108). It does not suppress data races or
lesson lock reports; the lock-order positive control verifies the latter. To inspect the original
external warning with that rule disabled, run the reporting test explicitly (expected nonzero):

```bash
python3 21_integrated_tensorrt_video_pipeline/tests/run_tsan.py --no-gdal-suppression -- \
  ./21_integrated_tensorrt_video_pipeline/build-tsan-clang18/integrated_reporting_tests \
  21_integrated_tensorrt_video_pipeline/build-tsan-clang18/reporting-test-output
```

Save verbose evidence under ignored `reports/` when needed:

```bash
mkdir -p reports
ctest --test-dir 21_integrated_tensorrt_video_pipeline/build-tsan-clang18 \
  --verbose --output-log reports/21_tsan.log
```

Local verification used `learn-tensorrt:25.11`, an NVIDIA GeForce RTX 4090, TensorRT 10.14.1,
CUDA runtime 13000, and CUDA driver API 13020. The full GPU suite (24 tests), CPU-only suite
(3 tests), default CPU/GPU commands, and a two-second duration stop were run successfully.
The duration check also observed live snapshots with `submitted > completed`. The Clang TSAN suite
above completed after the original GCC startup failure. The 1800-second example was not run as part
of this verification.

## Checkpoints

1. Why is a shared engine safe while a concurrently used execution context is not shared?
2. Which event releases a slot, and why is `cudaDeviceSynchronize()` not the steady-state answer?
3. Which work drains at EOS, and which work is discarded or quiesced after abort?
4. Why can an empty `cv::Mat` destination preserve queued pixels when a decoder reuses buffers?
5. During an abort, how do admission rejection, unsubmitted discard, and failed result dispatch differ?
6. Why must batch-fill waiting not be added to `queue_wait_ms`, and why is JPEG output timed separately?
