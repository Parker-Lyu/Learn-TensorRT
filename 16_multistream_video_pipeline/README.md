# 16 - Multi-Stream Video Pipeline

Goal: build a production-style inference pipeline for multiple cameras or video streams.

Topics:

- Multi-stream input configuration
- One capture thread per stream or a capture thread pool
- Per-stream bounded queues
- `stream_id` and `frame_id`
- Timestamp propagation
- Round-robin scheduling
- Latest-frame-first scheduling
- Micro-batching timeout
- Dynamic batch with partially filled batches
- Result dispatch back to the source stream
- Per-stream FPS, latency, queue depth, and dropped-frame metrics
- Graceful shutdown across multiple threads

Acceptance criteria:

- The program reads from at least two video files or camera-like sources.
- Each stream has independent FPS, queue depth, and dropped-frame counters.
- Frames are batched for TensorRT inference when possible.
- Detection results are routed back to the correct stream.
- The report includes total throughput and per-stream P50/P90/P99 latency.
- You can explain the trade-off between fairness, throughput, and real-time freshness.
