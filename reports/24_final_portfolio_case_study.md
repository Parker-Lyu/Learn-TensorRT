# Final TensorRT Deployment Portfolio Case Study

Repository evidence revision: `a66d7f3`. This project develops a YOLOv8n deployment from ONNX
export through TensorRT C++ inference, precision validation, bounded asynchronous pipelines, CUDA
preprocessing, server/edge integration exercises, and reusable language bindings.

## Evidence Map

- [10a functional and architecture validation](10a_end_to_end_validation.md)
- [12a precision and performance decision](12a_precision_performance.md)
- [17a pipeline performance and reliability](17a_pipeline_performance.md)

The linked reports retain commands and raw-artifact paths; this case study does not duplicate them.

## Verification Environment

- Development image: `learn-tensorrt:25.11`
- Container build identity: `231036167`
- Declared course GPU: `NVIDIA GeForce RTX 4090`
- Runtime GPU / compute capability / driver / memory MiB query:
  `NVIDIA GeForce RTX 4090, 8.9, 595.84, 24564`
- TensorRT: `10.14.1.48`
- CUDA Toolkit: `Build cuda_13.0.r13.0/compiler.36424714_0`

Performance and acceptance results are valid only for their recorded hardware and software identity.

## Precision Decision

| Precision | Samples | Mean ms | P50 ms | P90 ms | P99 ms | Images/s | Accuracy gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP32 | 120 | 1.222 | 1.134 | 1.473 | 1.934 | 1172.8 | PASS |
| FP16 | 120 | 0.861 | 0.817 | 0.888 | 1.515 | 2001.3 | PASS |
| INT8 | 120 | 1.005 | 0.959 | 0.998 | 1.695 | 1578.8 | PASS |

INT8 is the current deployment candidate under the declared gate. The decision uses the canonical 5,000-image human-labeled validation split
and identity-linked engine evidence. A one-input Polygraphy alignment is valuable for locating
numerical conversion bugs, but it cannot replace dataset accuracy, tail latency, or long-run
reliability.

## Pipeline Result

| Captured | Processed | Dropped | Queue peak | FPS | P50 ms | P90 ms | P99 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 496 | 4 | 4 | 813.01 | 11.18 | 12.49 | 13.06 |

The latest-frame bounded queue trades completeness for freshness under sustained overload. Normal
EOS drains work; cancellation and failure discard queued work and join all workers. Multi-stream
round-robin scheduling protects fairness, while dynamic batching trades queue delay for throughput.

## Reusable Structure

- C++ preprocessing, postprocessing, TensorRT runner, and visualization libraries in lesson 10.
- Generic bounded queues and lifecycle tests in lessons 13–16.
- CUDA/NPP preprocessing with pageable, pinned, and mapped-memory evidence in lesson 17.
- Stable C ABI plus Python ctypes wrapper in lesson 21.
- Focused algorithm and RAII tests in lesson 23.

## Test Evidence

| Check | Result |
| --- | --- |
| lesson10 preprocessing/postprocessing | PASS |
| lesson13 concurrency | PASS |
| lesson14 batching | PASS |
| lesson15 async pipeline | PASS |
| lesson16 multistream | PASS |
| lesson17 CUDA preprocess | PASS |
| lesson21 ctypes inference | PASS |
| lesson23 katas | PASS |

Local-equivalent status: **PASS**. The JSON evidence retains each
command's stdout, stderr, return code, duration, and platform query. A failed or unavailable GPU check
remains failed rather than being converted into a CPU-only pass. The formal 30-minute soak and a
successful ThreadSanitizer run remain separate release gates.

## Delivery Image

`24_final_portfolio_case_study/Dockerfile` uses the pinned development image as its builder and a
CUDA 13.0 Ubuntu 24.04 base as its runtime stage. The runtime contains the executable, TensorRT 10
runtime library, OpenCV runtime packages, one engine, and `assets/img.jpeg`. Engines must be rebuilt
for the deployment environment.

| Image | ID | Size MiB |
| --- | --- | ---: |
| `learn-tensorrt:25.11, learn-tensorrt:25.11-audit` | `c231b081df78` | 9308.2 |
| `nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04` | `6e43a6b02e5f` | 132.9 |
| `learn-tensorrt-runtime:10.14` | `4bafbc73d5e3` | 568.4 |

## Completed Elective Track

The advanced NVIDIA deployment path includes Triton configuration/client load tests, ONNX graph
surgery, a real custom TensorRT CUDA plugin, DeepStream multi-stream configuration/parser work,
Jetson/DLA target procedures, a C ABI Python integration, and LLM inference awareness. Triton,
DeepStream, and Jetson runtime acceptance remains explicitly hardware/container dependent.

## Bottleneck and Future Work

Nsight evidence identified CPU preprocessing/postprocessing as the original end-to-end bottleneck.
CUDA/NPP reduced preprocessing work, but transfer strategy still matters. Next work is to
confirm INT8 behavior on the target deployment hardware, complete the formal soak/TSAN gates, and validate runtime behavior on
Triton, DeepStream, and Jetson hardware.

## Resume Bullets

- Built and validated a modular C++17 YOLOv8 TensorRT pipeline with FP32/FP16/INT8 release gates,
  CUDA/NPP preprocessing, dynamic batching, bounded multi-stream scheduling, and fault injection.
- Implemented a serialized CUDA TensorRT plugin, ONNX graph repair workflow, C ABI shared library,
  Python integration, and reproducible latency/accuracy/report generation.
- Profiled CPU/GPU bottlenecks and documented honest deployment boundaries for Triton, DeepStream,
  Jetson DLA, OpenVINO CPU, and local autoregressive LLM inference.

## Five-Minute English Presentation

Start with the deployment goal and controlled YOLOv8 model. Explain ONNX/TensorRT correctness and
why raw alignment precedes detection metrics. Present the measured precision decision from the 12a
gate. Walk through the bounded single/multi-stream design and capture-to-result tail latency. Show
the CUDA preprocessing and custom plugin evidence. Finish with reproducibility,
remaining soak/TSAN/hardware gates, and why those limitations are reported rather than hidden.

## Longer Interview Walkthrough

Discuss RAII ownership, dynamic profiles, tensor offsets, decode/NMS coordinate transforms, queue
close semantics, exception propagation, scheduling fairness, CUDA streams and transfer modes,
plugin registration/serialization, C ABI ownership, and deployment image contents. For every number,
trace it to the linked generated report and raw artifact; if evidence is incomplete, state the exact
command and environment needed to complete it.
