# 22 - Pipeline Performance and Reliability Report

## Purpose

This checkpoint converts lessons 16–21 into reproducible load, latency, fairness, memory, fault,
and sanitizer evidence. It generates `reports/22_pipeline_performance.md`; it does not silently
mark a short smoke run as a completed 30-minute soak.

## Prerequisites

- Build Lesson 20's CPU/CUDA preprocessing comparison and Lesson 21's integrated executable.
- Reproduce the Lesson 17 dynamic batch 1--4 engine using its documented commands.
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

```

<details><summary>Example output (local run)</summary>

```text
wrote /workspace/Learn-TensorRT/22_pipeline_performance_report/outputs/evidence.json
wrote /workspace/Learn-TensorRT/reports/22_pipeline_performance.md
```
</details>
bash
python3 22_pipeline_performance_report/collect_pipeline_evidence.py \
  --soak-minutes 30 \
  --restart-cycles 100 \
  --run-sanitizers
python3 22_pipeline_performance_report/generate_report.py
```

The default command is deliberately short: it runs three restart cycles, a three-second
single-process soak, the real Lesson 21 batch/policy/multi-stream matrix, reference checks, and the
integrated fault matrix. Formal soak, restart, and sanitizer gates remain `INCOMPLETE`.

The formal command keeps one Lesson 21 process alive for 30 minutes, separately launches 100 short
processes, samples the lesson PID's RSS/device memory, and runs direct Lesson 21 sanitizer checks.
Evidence records the pinned image, GPU, compute capability, driver, CUDA, and TensorRT identity.
Raw evidence and generated reports remain ignored and must be regenerated for each platform.

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

On kernels where the default container blocks `personality(2)` and GCC ThreadSanitizer reports
`unexpected memory mapping`, run the already-built CPU-only test in a disposable container with
only the seccomp restriction relaxed. The helper runs both Lesson 21 CPU test binaries and saves
machine-readable ignored evidence consumed by the main collector:

```bash
python3 22_pipeline_performance_report/collect_tsan_evidence.py
```

Do not mark TSAN complete unless that command actually returns zero without a sanitizer report.
