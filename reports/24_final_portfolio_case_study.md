# Final TensorRT Deployment Portfolio Case Study

Repository evidence revision: `a517b02`. This project develops a YOLOv8n deployment from ONNX
export through TensorRT C++ inference, precision validation, bounded asynchronous pipelines, CUDA
preprocessing, server/edge integration exercises, and reusable language bindings.

## Evidence Map

- [10a functional and architecture validation](10a_end_to_end_validation.md)
- [12a precision and performance decision](12a_precision_performance.md)
- [17a pipeline performance and reliability](17a_pipeline_performance.md)

The linked reports retain commands and raw-artifact paths; this case study does not duplicate them.

## Precision Decision

| Precision | Samples | Mean ms | P50 ms | P90 ms | P99 ms | Images/s | Accuracy gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP32 | 120 | 4.421 | 4.416 | 4.455 | 4.486 | 226.2 | PASS |
| FP16 | 120 | 2.726 | 2.725 | 2.742 | 2.761 | 366.9 | PASS |
| INT8 | 120 | 2.451 | 2.446 | 2.465 | 2.571 | 407.9 | FAIL |

FP16 is the current deployment choice. INT8 is faster but fails the predeclared smoke-set detection
gate; it remains blocked pending representative calibration/validation data, sensitive-layer
fallback, or QAT. A one-input Polygraphy alignment is valuable for locating numerical conversion
bugs, but it cannot establish dataset accuracy, tail latency, or long-run reliability.

## Pipeline Result

| Captured | Processed | Dropped | Queue peak | FPS | P50 ms | P90 ms | P99 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 286 | 214 | 4 | 626.38 | 12.59 | 15.28 | 20.08 |

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
| lesson13 concurrency | PASS |
| lesson14 batching | PASS |
| lesson15 async pipeline | PASS |
| lesson16 multistream | PASS |
| lesson17 CUDA preprocess | PASS |
| lesson21 ctypes | PASS |
| lesson23 katas | PASS |

Local-equivalent status: **PASS**. GPU/TensorRT checks run in the
pinned development environment. Current unresolved evidence remains the full 30-minute soak and a
host where ThreadSanitizer starts successfully.

## Delivery Image

`24_final_portfolio_case_study/Dockerfile` uses a TensorRT development builder and a CUDA runtime
stage containing the executable, required TensorRT runtime library, OpenCV runtime packages, one
engine, and one input fixture. Engines must be rebuilt for the deployment environment.

Image-size evidence: `Not measured: Docker unavailable on this host.`

## Completed Elective Track

The advanced NVIDIA deployment path includes Triton configuration/client load tests, ONNX graph
surgery, a real custom TensorRT CUDA plugin, DeepStream multi-stream configuration/parser work,
Jetson/DLA target procedures, a C ABI Python integration, and LLM inference awareness. Triton,
DeepStream, and Jetson runtime acceptance remains explicitly hardware/container dependent.

## Bottleneck and Future Work

Nsight evidence identified CPU preprocessing/postprocessing as the original end-to-end bottleneck.
CUDA/NPP reduced preprocessing work, but transfer strategy still matters. Next work is a real labeled
validation set, accepted INT8 calibration or QAT, the formal soak/TSAN gates, and runtime validation
on Triton, DeepStream, and Jetson hardware.

## Resume Bullets

- Built and validated a modular C++17 YOLOv8 TensorRT pipeline with FP32/FP16/INT8 release gates,
  CUDA/NPP preprocessing, dynamic batching, bounded multi-stream scheduling, and fault injection.
- Implemented a serialized CUDA TensorRT plugin, ONNX graph repair workflow, C ABI shared library,
  Python integration, and reproducible latency/accuracy/report generation.
- Profiled CPU/GPU bottlenecks and documented honest deployment boundaries for Triton, DeepStream,
  Jetson DLA, OpenVINO CPU, and local autoregressive LLM inference.

## Five-Minute English Presentation

Start with the deployment goal and controlled YOLOv8 model. Explain ONNX/TensorRT correctness and
why raw alignment precedes detection metrics. Present FP16 as the accepted precision and INT8 as a
failed accuracy gate. Walk through the bounded single/multi-stream design and capture-to-result tail
latency. Show the CUDA preprocessing and custom plugin evidence. Finish with reproducibility,
remaining soak/TSAN/hardware gates, and why those limitations are reported rather than hidden.

## Longer Interview Walkthrough

Discuss RAII ownership, dynamic profiles, tensor offsets, decode/NMS coordinate transforms, queue
close semantics, exception propagation, scheduling fairness, CUDA streams and transfer modes,
plugin registration/serialization, C ABI ownership, and deployment image contents. For every number,
trace it to the linked generated report and raw artifact; if evidence is incomplete, state the exact
command and environment needed to complete it.
