# 15 - Async Video Pipeline

Goal: move from single-image inference to a production-like video pipeline.

Topics:

- Video input
- Frame queue
- Producer-consumer queue integration
- Pinned memory
- Double buffering
- Async inference
- `enqueueV3` or the TensorRT enqueue API used by the local TensorRT version
- CPU/GPU overlap
- Dynamic batching from queued frames
- Frame timestamp tracking
- Dropped-frame statistics
- FPS and latency percentiles

Acceptance criteria:

- The program processes a video file or camera stream.
- Average FPS, P50/P90/P99 latency, and GPU utilization are recorded.
