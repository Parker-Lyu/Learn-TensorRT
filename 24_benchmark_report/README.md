# 24 - Benchmark Report

Goal: turn the learning project into interview-ready evidence.

Deliverables:

- `report.md`
- Environment table
- Model export notes
- FP32, FP16, and INT8 latency table
- Throughput table
- Single-input precision-alignment note from lesson 06a
- Multi-image accuracy-regression notes from lesson 12
- Detection-quality examples for any changed FP16 or INT8 outputs
- Profiler timeline notes
- Per-stream metrics when multi-stream tests are available
- Test evidence table
- CI/build notes
- Bottleneck analysis
- Production Docker packaging
- Multi-stage Dockerfile
- Development image versus runtime image size comparison
- Future work

Acceptance criteria:

- A recruiter or interviewer can understand the project in five minutes.
- The report explains what was measured, how it was measured, and what the numbers mean.
- You can defend every number in the report.
- Accuracy evidence separates raw tensor alignment, multi-image drift statistics, and decoded
  detection-quality checks.
- The final report points to the reusable module structure and the tests that protect core preprocessing, postprocessing, and resource-management behavior.
- CI or a documented local equivalent configures, builds, and runs the available tests.
- A production-style Docker image can run the final executable or shared library without build tools.
- The report compares development image size and runtime image size.
- A multi-stage Docker build produces a runtime image that contains only the files needed to run inference.
