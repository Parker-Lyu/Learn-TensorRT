# 15 - Async Single-Stream Video Pipeline

This lesson turns the bounded queue from lesson 13 into a production-shaped single-stream pipeline:
capture, latest-frame backpressure, timeout-based micro-batching, two in-flight worker slots, and
capture-to-result latency metrics.

## Build and Run

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/async_video_pipeline
```

The default synthetic source makes the artifact runnable without a camera. Use a video or camera:

```bash
./build/async_video_pipeline --input video.mp4
./build/async_video_pipeline --input 0
```

Important controls are `--queue-capacity`, `--max-batch`, `--batch-timeout-ms`, and
`--inference-ms`. The last option is a deterministic stand-in for TensorRT compute so concurrency,
shutdown, overload, and metrics remain CPU-testable. Replace that task with lesson 14's dynamic
runner without changing capture or queue ownership.

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

## Failure Experiments

```bash
./build/async_video_pipeline --fail-capture-at 20
./build/async_video_pipeline --fail-worker-at 20
./build/async_video_pipeline --input does-not-exist.mp4
```

Each returns nonzero, closes the queue, joins both threads, and propagates the original error. Tests
also verify bounded overload accounting and explicit stop without deadlock.
