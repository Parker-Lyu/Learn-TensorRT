# 23 - C++ Deployment Interview Katas

## Purpose

This lesson collects compact C++17 implementations that commonly appear in CV deployment
interviews: IoU, class-aware NMS, bilinear sampling, letterbox coordinate mapping, HWC-to-CHW,
stable Top-K, a ring buffer, a closeable bounded queue, and a move-only CUDA buffer.

## Prerequisites

- Use the pinned development container for the complete C++ and CUDA test set.
- CPU-only katas remain runnable when a CUDA device is unavailable.

## Deliverables

- Reusable C++17 kata library and demo executable
- Focused CPU algorithm, queue, ring-buffer, and CUDA ownership tests
- Documented practice timing tied to earlier lessons

## Build

Configure and build from the repository root inside the pinned development container:

```bash
cmake -S 23_cpp_interview_katas -B 23_cpp_interview_katas/build
cmake --build 23_cpp_interview_katas/build --parallel
```

The generated build directory is ignored.

## Run

```bash
./23_cpp_interview_katas/build/katas_demo
```

Tests cover empty NMS, degenerate boxes, overlapping boxes from different classes, interpolation
boundaries, extreme letterbox clamping, Top-K ties, ring wrap/full/empty behavior, queue close waking
a blocked producer, and CUDA ownership transfer. The CPU katas and CUDA ownership checks are separate
CTest cases, so a machine without an accessible CUDA device reports only the CUDA case as skipped.

Practice rewriting one group from memory after its related core lesson. Explain input validation,
ownership, synchronization, and algorithmic complexity before optimizing syntax.

## Outputs

- `build/katas_demo` is the ignored runnable artifact.
- CTest reports CPU and CUDA cases separately so CUDA unavailability is not hidden.

## Tests

Run the configured CTest suite:

```bash
ctest --test-dir 23_cpp_interview_katas/build --output-on-failure
```

## Checkpoints

1. Implement deployment-relevant C++ algorithms and ownership patterns without framework wrappers.
2. Explain validation, complexity, boundary behavior, ownership, and synchronization choices.
3. Use focused tests to practice each kata after its related course lesson.
