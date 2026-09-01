# 19 - Multi-Stream Video Pipeline

## Purpose

This lesson scales the single-stream design to independent capture threads and bounded queues,
global scheduling, dynamic micro-batches, out-of-order async completion, and identity-safe result
dispatch.

## Prerequisites

- Complete lessons 16 and 18.
- The default synthetic sources need no camera or video file.

## Deliverables

- Reusable multi-stream scheduler and pipeline library
- Runnable multi-source executable
- Identity, fairness, overload, shutdown, and failure-policy tests

## Architecture

```text
capture 0 -> bounded queue 0 --+
capture 1 -> bounded queue 1 --+-> scheduler -> micro-batch -> two async slots -> dispatcher
capture N -> bounded queue N --+                                  | stream_id + frame_id
```

`round-robin` favors fairness. `latest` removes stale queued frames and favors freshness. A batch is
submitted when full or after the timeout, so a slow stream cannot indefinitely block a fast one.
The async worker deliberately varies completion delay; the dispatcher uses immutable
`stream_id/frame_id` rather than completion order.

## Failure Policy

The default source policy isolates a failed source: its queue closes while healthy streams continue.
`StopAll` is available in the library for applications where any missing camera invalidates the
whole result. Inference failure always stops all streams because shared result correctness can no
longer be guaranteed.

## Build

Configure and build from the repository root inside the pinned development container:

```bash
cmake -S 19_multistream_video_pipeline -B 19_multistream_video_pipeline/build
cmake --build 19_multistream_video_pipeline/build --parallel
```

The generated build directory is ignored.

## Run

Run the commands from the repository root:

```bash
./19_multistream_video_pipeline/build/multistream_video_pipeline
```

<details><summary>Example output (local run)</summary>

```text
total_fps=610.03
stream=0 captured=120 processed=120 dropped=0 queue_peak=1 fps=366.02 p50=7.22 p90=9.29 p99=10.21
stream=1 captured=80 processed=80 dropped=0 queue_peak=1 fps=244.01 p50=6.09 p90=8.54 p99=10.37
```
</details>

The default uses two camera-like synthetic streams with different rates. Real files are repeatable:

```bash
./19_multistream_video_pipeline/build/multistream_video_pipeline --input camera-a.mp4 --input camera-b.mp4
```

### Metrics and Experiments

The program reports total throughput and per-stream captured, processed, dropped, queue peak, FPS,
and capture-to-result P50/P90/P99 latency.

1. Compare `--scheduler round-robin` and `--scheduler latest` under unequal input rates.
2. Increase batch size and explain throughput versus the slow stream's tail latency.
3. Run `--fail-inference-batch 2` and verify a nonzero exit without blocked capture threads.
4. Inspect the integrity test: every output identity must be unique and belong to its source even
   when batches complete out of order.

## Outputs

- The executable reports total throughput and per-stream capture, processing, dropping, queue, FPS,
  and percentile-latency metrics.
- The current deterministic worker produces scheduling evidence, not TensorRT detection output.

## Tests

Run the configured CTest suite:

```bash
ctest --test-dir 19_multistream_video_pipeline/build --output-on-failure
```

## Checkpoints

1. Schedule and batch frames from multiple independently bounded streams.
2. Preserve stream and frame identity through batching and out-of-order result completion.
3. Compare fairness, throughput, latency, and freshness under overload and source failures.
