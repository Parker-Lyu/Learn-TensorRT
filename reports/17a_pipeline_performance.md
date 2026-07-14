# 17a - Pipeline Performance and Reliability Report

Generated from saved measurements. Checkpoint status: **INCOMPLETE**.

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
| 500 | 286 | 214 | 4 | 626.38 | 12.59 | 15.28 | 20.08 |

Capture timestamps flow through the queue and async result collection, so these are end-to-end
capture-to-result latencies rather than model-only timings.

Host RSS MiB start/peak/end: 0.02 / 75.26 / 75.24.
Device memory MiB start/peak/end: 0.00 / 0.00 / 0.00.

## Multi-Stream Fairness

Total throughput: 607.66 frames/s.

| Stream | Captured | Processed | Dropped | FPS | P50 ms | P90 ms | P99 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 120 | 120 | 0 | 364.60 | 7.34 | 9.37 | 9.69 |
| 1 | 80 | 80 | 0 | 243.06 | 6.96 | 9.33 | 10.53 |

Round-robin prevents a fast source from monopolizing every batch. Larger batches improve worker
efficiency but can increase timeout wait and tail latency; latest-first reduces stale work but may
drop more frames from bursty streams.

## CPU vs CUDA/NPP Preprocessing

| Memory mode | CPU ms | Host stage ms | H2D ms | GPU/NPP ms | D2H ms | Mean abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pageable | 3.579 | 0.000 | 0.692 | 0.066 | 1.105 | 0.01585 |
| pinned | 3.579 | 0.619 | 0.407 | 0.065 | 0.734 | 0.01585 |
| mapped | 3.579 | 0.584 | 0.003 | 1.340 | 0.002 | 0.01585 |

Mapped memory removes explicit transfers but makes a discrete GPU access host memory across PCIe;
the measured kernel time must be included before describing it as faster.

## Lifecycle, Faults, and Sanitizers

- Repeated start/stop: 100 cycles, 0 failures.
- Soak requested: 0.020 minutes across 2 cycles, 0 failures.

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

ThreadSanitizer output: `FATAL: ThreadSanitizer: unexpected memory mapping 0x6100ab163000-0x6100ab168000`.
The current host may reject TSAN before tests start with an unexpected memory mapping; that is an
environment limitation, not a passing race check. Run the pinned container/host combination where
TSAN starts successfully before marking this gate complete.

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
