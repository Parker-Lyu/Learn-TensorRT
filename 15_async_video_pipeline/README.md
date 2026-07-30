# 15 - Async Single-Stream Video Pipeline

## Purpose

This lesson turns the bounded queue from lesson 13 into a production-shaped single-stream pipeline:
capture, latest-frame backpressure, timeout-based micro-batching, two in-flight worker slots, and
capture-to-result latency metrics.

## Prerequisites

- Complete lesson 13.
- OpenCV video support is required for file or camera input; the default synthetic source needs no external media.

## Deliverables

- Reusable asynchronous single-stream pipeline library
- Runnable video-pipeline executable
- Lifecycle, overload, end-of-stream, and failure-path tests

## Semantics and Metrics

- Sustained overload drops the oldest queued frame to favor real-time freshness.
- Normal end-of-stream closes in drain mode; cancellation and failures discard queued work.
- Two async tasks form the double buffer: one batch can execute while the next is assembled.
- A batch is submitted when full or when `batch-timeout-ms` expires.
- Reported latency begins at capture and ends when the async result is collected.
- The executable reports FPS, P50/P90/P99 latency, queue peak, processed, and dropped frames.

Sample GPU utilization beside a real TensorRT backend with:

```bash
nvidia-smi dmon -s u -d 1 -o DT > outputs/gpu_utilization.log
```

Do not label the simulated worker's CPU timing as GPU utilization. GPU utilization must come from
the real backend run and the monitoring capture above.

## Build

Configure and build from the repository root inside the pinned development container:

```bash
cmake -S 15_async_video_pipeline -B 15_async_video_pipeline/build
cmake --build 15_async_video_pipeline/build --parallel
```

The generated build directory is ignored.

## Run

Run the commands from the repository root:

```bash
./15_async_video_pipeline/build/async_video_pipeline
```

The default synthetic source makes the artifact runnable without a camera. Use a video or camera:

```bash
./15_async_video_pipeline/build/async_video_pipeline --input video.mp4
./15_async_video_pipeline/build/async_video_pipeline --input 0
```

Important controls are `--queue-capacity`, `--max-batch`, `--batch-timeout-ms`, and
`--inference-ms`. The last option is a deterministic stand-in for TensorRT compute so concurrency,
shutdown, overload, and metrics remain CPU-testable. Replace that task with lesson 14's dynamic
runner without changing capture or queue ownership.

### Failure Experiments

```bash
./15_async_video_pipeline/build/async_video_pipeline --fail-capture-at 20
./15_async_video_pipeline/build/async_video_pipeline --fail-worker-at 20
./15_async_video_pipeline/build/async_video_pipeline --input does-not-exist.mp4
```

Each returns nonzero, closes the queue, joins both threads, and propagates the original error. Tests
also verify bounded overload accounting and explicit stop without deadlock.

## Outputs

- The executable reports FPS, P50/P90/P99 capture-to-result latency, queue peak, processed frames, and dropped frames.
- Optional monitoring logs belong under ignored `outputs/`; simulated worker timing is not GPU evidence.

## Tests

Run the configured CTest suite:

```bash
ctest --test-dir 15_async_video_pipeline/build --output-on-failure
```

## Checkpoints

1. Implement a bounded asynchronous single-stream pipeline with explicit ownership and cancellation.
2. Measure capture-to-result latency, throughput, queue depth, and dropped-frame behavior.
3. Handle end-of-stream, invalid input, overload, and worker failure without deadlock.
