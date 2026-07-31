# 21 - Integrated TensorRT Video Pipeline

## Purpose

Integrate the bounded multi-stream scheduling semantics from lessons 16–19 with lesson 20 CUDA/NPP
preprocessing and real TensorRT dynamic execution. This lesson establishes correctness, ownership,
and measurement boundaries; lesson 22 owns formal soak and performance conclusions.

## Prerequisites

Use the pinned development container. Build the dynamic batch 1–4 engine documented by lesson 17;
do not reuse an undocumented local engine. `assets/img.jpeg` is the default controlled image.

## Deliverables

- `integrated_pipeline_core`: CPU-testable immutable metadata, slot lifecycle, accounting, and dispatch.
- `integrated_tensorrt_video_pipeline`: runnable identity/ownership smoke executable.
- Focused CPU tests. The asynchronous CUDA/NPP-to-TensorRT backend is the GPU acceptance boundary.

## Design

The shared engine outlives every slot. Each slot exclusively owns its context, stream, lifecycle
completion event, timing events, and buffers. The submitter performs `Free -> Reserved -> Submitted`;
the collector performs `Submitted -> Completing -> Free`. Completion order never defines identity.
Normal EOS drains. Abort rejects new work, discards unsubmitted work, and quiesces submitted CUDA
work before resource destruction; already submitted CUDA work is not described as cancellable.

Counters distinguish captured, admitted, admission rejection, queue eviction, submission,
completion, in-flight failure, and abort discard. CUDA events measure GPU stages; the monotonic host
clock measures queueing and capture-to-dispatch latency.

## Build

```bash
cmake -S 21_integrated_tensorrt_video_pipeline -B 21_integrated_tensorrt_video_pipeline/build
cmake --build 21_integrated_tensorrt_video_pipeline/build --parallel
```

## Run

CPU ownership and reverse-completion smoke test:

```bash
./21_integrated_tensorrt_video_pipeline/build/integrated_tensorrt_video_pipeline
```

This smoke path is explicitly not TensorRT or performance evidence. Formal GPU acceptance requires
the dynamic engine and the pinned GPU container.

## Outputs

Structured records and annotated samples belong under ignored `output/`. Source, tests, and this
README are committed; engines, measurements, and generated images are not.

## Tests

```bash
ctest --test-dir 21_integrated_tensorrt_video_pipeline/build --output-on-failure
```

These focused tests are CPU-only and force reverse completion to validate slot transitions,
terminal accounting, metadata propagation, and identity dispatch. GPU integration checks require
the pinned container, TensorRT engine, and NVIDIA GPU.

## Checkpoints

1. Why is a shared engine safe while a concurrently used execution context is not shared?
2. Which event releases a slot, and why is `cudaDeviceSynchronize()` not the steady-state answer?
3. Which work drains at EOS, and which work is discarded or quiesced after abort?
