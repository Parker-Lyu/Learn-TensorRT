# 17a - Pipeline Performance and Reliability Report

Generated from saved measurements. Checkpoint status: **INCOMPLETE**.

## Measurement Environment

- Development image: `nvcr.io/nvidia/pytorch:25.11-py3`
- GPU / compute capability / driver / memory MiB: `NVIDIA GeForce RTX 4090, 8.9, 595.71.05, 24564`
- TensorRT: `10.14.1.48`

Performance values are valid only for this recorded software and hardware environment.

## Architecture

```text
single stream:
capture -> bounded latest-frame queue -> timeout batcher -> two async slots -> latency metrics

multi stream:
capture 0 -> queue 0 --+
capture 1 -> queue 1 --+-> fair scheduler -> partial batch -> async worker -> ID dispatcher
capture N -> queue N --+
```

Queues drop the oldest frame under sustained overload to preserve freshness. Normal EOS drains;
cancellation and worker failure discard queued work. Round-robin is the measured fairness policy;
latest-first remains available when freshness matters more than equal service.

## Single-Stream Load

| Captured | Processed | Dropped | Queue peak | FPS | P50 ms | P90 ms | P99 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 496 | 4 | 4 | 813.01 | 11.18 | 12.49 | 13.06 |

Capture timestamps flow through the queue and async result collection, so these are end-to-end
capture-to-result latencies rather than model-only timings.

Host RSS MiB start/peak/end: 0.20 / 77.95 / 77.95.
Device memory MiB start/peak/end: 0.00 / 0.00 / 0.00.

## Multi-Stream Fairness

Total throughput: 601.45 frames/s.

| Stream | Captured | Processed | Dropped | FPS | P50 ms | P90 ms | P99 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 120 | 120 | 0 | 360.87 | 7.26 | 9.34 | 10.25 |
| 1 | 80 | 80 | 0 | 240.58 | 7.53 | 9.32 | 10.33 |

Round-robin prevents a fast source from monopolizing every batch. Larger batches improve worker
efficiency but can increase timeout wait and tail latency; latest-first reduces stale work but may
drop more frames from bursty streams.

## CPU vs CUDA/NPP Preprocessing

| Memory mode | CPU ms | Host stage ms | H2D ms | GPU/NPP ms | D2H ms | Mean abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pageable | 1.298 | 0.000 | 0.158 | 0.013 | 0.329 | 0.01585 |
| pinned | 1.298 | 0.117 | 0.140 | 0.014 | 0.245 | 0.01585 |
| mapped | 1.298 | 0.117 | 0.001 | 0.478 | 0.001 | 0.01585 |

Mapped memory removes explicit transfers but makes a discrete GPU access host memory across PCIe;
the measured kernel time must be included before describing it as faster.

## Lifecycle, Faults, and Sanitizers

- Repeated start/stop: 100 cycles, 0 failures.
- Soak requested: 0.020 minutes across 1 cycles, 0 failures.

| Fault | Return code | Expected nonzero |
| --- | ---: | --- |
| invalid_input | 1 | PASS |
| capture_failure | 1 | PASS |
| worker_failure | 1 | PASS |
| multistream_inference_failure | 1 | PASS |

| Gate | Status |
| --- | --- |
| bounded single | PASS |
| single accounting | PASS |
| bounded multi | PASS |
| restart 100 | PASS |
| soak 30 minutes | NOT COMPLETE |
| fault matrix | PASS |
| compute sanitizer | PASS |
| thread sanitizer | NOT COMPLETE |

ThreadSanitizer output: `FATAL: ThreadSanitizer: unexpected memory mapping 0x5bc20de7f000-0x5bc20de85000`.
A nonzero ThreadSanitizer return code keeps the gate incomplete; never reinterpret a tool startup
failure as a passing race check.

## Reproduction

Smoke collection:

```bash
python3 17a_pipeline_performance_report/collect_pipeline_evidence.py
python3 17a_pipeline_performance_report/generate_report.py
```

Formal checkpoint collection:

```bash
python3 17a_pipeline_performance_report/collect_pipeline_evidence.py --soak-minutes 30 --restart-cycles 100
python3 17a_pipeline_performance_report/generate_report.py
```

## English Summary

The pipeline uses bounded latest-frame queues, timeout-based batching, and explicit drain or discard
shutdown. Single- and multi-stream measurements report capture-to-result percentiles instead of
average FPS alone. Identity tests protect stream and frame routing under asynchronous completion.
CUDA/NPP preprocessing is numerically compared with OpenCV and transfer costs remain separate.
The checkpoint stays incomplete until the full thirty-minute soak and a runnable ThreadSanitizer
environment both pass.

## Three-to-Five-Minute Walkthrough

Describe the queue and scheduler diagrams, then show how overload bounds memory while trading frame
completeness for freshness. Compare total throughput with per-stream tail latency. Explain the
pageable, pinned, and mapped preprocessing results, then finish with repeated lifecycle, injected
failures, sanitizer evidence, and any remaining incomplete gates.
