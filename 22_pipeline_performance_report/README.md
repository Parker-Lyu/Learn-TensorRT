# 22 - Pipeline Performance and Reliability Report

## Purpose

This checkpoint converts lessons 16–21 into reproducible load, latency, fairness, memory, fault,
and sanitizer evidence. It generates `reports/22_pipeline_performance.md`; it does not silently
mark a short smoke run as a completed 30-minute soak.

Build lessons 16, 18, 19, 20, and 21 first, then run from the repository root:

## Prerequisites

- Build lessons 16, 18, 19, 20, and 21 first.
- Formal acceptance requires the documented GPU, sanitizer, soak, and restart environment.

## Deliverables

- `collect_pipeline_evidence.py` load and reliability evidence collector
- `generate_report.py` report generator and focused tests
- `reports/22_pipeline_performance.md` generated checkpoint report

## Generate the Report

```bash
python3 22_pipeline_performance_report/collect_pipeline_evidence.py
python3 22_pipeline_performance_report/generate_report.py
```

For the formal checkpoint:

```bash
python3 22_pipeline_performance_report/collect_pipeline_evidence.py \
  --soak-minutes 30 \
  --restart-cycles 100
python3 22_pipeline_performance_report/generate_report.py
```

The collector runs 100 lifecycle cycles, four fault cases, CUDA memcheck, and the available TSAN
binary. Host RSS and device memory are sampled around the single-stream load. The evidence also
records
the pinned development image, RTX GPU identity, compute capability, driver, memory, and TensorRT
version so performance results remain tied to the measured platform. Both raw evidence and the
generated report remain ignored and must be regenerated for the current platform.

## Outputs

- Raw load, memory, fault, restart, and sanitizer evidence is written under ignored `outputs/`.
- The generated `reports/22_pipeline_performance.md` is ignored and must state incomplete formal gates explicitly.

## Tests

Run the Python tests from the repository root:

```bash
python3 -m unittest discover -s 22_pipeline_performance_report/tests -v
```

## Checkpoints

1. Generate reproducible load, latency, throughput, fairness, memory, and stability evidence for lessons 16 through 21.
2. Evaluate overload and failure policies with soak, restart, sanitizer, and fault-injection results.
3. Explain the measured trade-offs among batching efficiency, per-stream fairness, and real-time freshness.
