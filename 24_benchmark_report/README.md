# 24 - Benchmark Report

Goal: turn the learning project into interview-ready evidence.

Deliverables:

- `report.md`
- Environment table
- Model export notes
- FP32, FP16, and INT8 latency table
- Throughput table
- Accuracy notes
- Profiler timeline notes
- Per-stream metrics when multi-stream tests are available
- Bottleneck analysis
- Production Docker packaging
- Multi-stage Dockerfile
- Runtime image size comparison
- Future work

Acceptance criteria:

- The report explains what was measured, how it was measured, and what the numbers mean.
- A production-style Docker image can run the final executable or shared library without build tools.
- The report compares development image size and runtime image size.
