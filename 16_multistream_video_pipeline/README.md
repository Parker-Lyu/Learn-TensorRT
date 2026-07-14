# 16 - Multi-Stream Video Pipeline

This lesson scales the single-stream design to independent capture threads and bounded queues,
global scheduling, dynamic micro-batches, out-of-order async completion, and identity-safe result
dispatch.

## Build and Run

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/multistream_video_pipeline
```

The default uses two camera-like synthetic streams with different rates. Real files are repeatable:

```bash
./build/multistream_video_pipeline --input camera-a.mp4 --input camera-b.mp4
```

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

## Metrics and Experiments

The program reports total throughput and per-stream captured, processed, dropped, queue peak, FPS,
and capture-to-result P50/P90/P99 latency.

1. Compare `--scheduler round-robin` and `--scheduler latest` under unequal input rates.
2. Increase batch size and explain throughput versus the slow stream's tail latency.
3. Run `--fail-inference-batch 2` and verify a nonzero exit without blocked capture threads.
4. Inspect the integrity test: every output identity must be unique and belong to its source even
   when batches complete out of order.
