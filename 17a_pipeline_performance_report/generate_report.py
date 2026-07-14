#!/usr/bin/env python3
"""Render checkpoint 17a from collected evidence and lesson 17 CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(evidence: dict) -> dict:
    single = evidence["single_stream"]["metrics"]
    multi = evidence["multi_stream"]["metrics"]
    return {
        "bounded_single": single["queue_peak"] <= 4,
        "single_accounting": single["captured"] == single["processed"] + single["dropped"],
        "bounded_multi": all(stream["queue_peak"] <= 4 for stream in multi["streams"]),
        "restart_100": evidence["restarts"]["requested"] >= 100 and evidence["restarts"]["failures"] == 0,
        "soak_30_minutes": evidence["soak"]["requested_minutes"] >= 30 and evidence["soak"]["failures"] == 0,
        "fault_matrix": all(item["expected_nonzero"] for item in evidence["faults"].values()),
        "compute_sanitizer": evidence["sanitizers"]["compute_memcheck"]["returncode"] == 0,
        "thread_sanitizer": evidence["sanitizers"]["thread_sanitizer"]["returncode"] == 0,
    }


def render(evidence: dict, cuda_rows: list[dict]) -> str:
    gates = evaluate(evidence)
    overall = "PASS" if all(gates.values()) else "INCOMPLETE"
    single = evidence["single_stream"]["metrics"]
    multi = evidence["multi_stream"]["metrics"]
    memory = evidence["single_stream"]
    gate_rows = "\n".join(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'NOT COMPLETE'} |"
        for name, passed in gates.items())
    stream_rows = "\n".join(
        f"| {int(stream['stream'])} | {int(stream['captured'])} | {int(stream['processed'])} | "
        f"{int(stream['dropped'])} | {stream['fps']:.2f} | {stream['p50']:.2f} | "
        f"{stream['p90']:.2f} | {stream['p99']:.2f} |"
        for stream in multi["streams"]
    )
    cuda_table = "\n".join(
        f"| {row['mode']} | {float(row['cpu_ms']):.3f} | {float(row['host_stage_ms']):.3f} | "
        f"{float(row['h2d_ms']):.3f} | {float(row['gpu_preprocess_ms']):.3f} | "
        f"{float(row['d2h_ms']):.3f} | {float(row['mean_abs_error']):.5f} |"
        for row in cuda_rows
    )
    fault_rows = "\n".join(
        f"| {name} | {item['returncode']} | {'PASS' if item['expected_nonzero'] else 'FAIL'} |"
        for name, item in evidence["faults"].items()
    )
    return f"""# 17a - Pipeline Performance and Reliability Report

Generated from saved measurements. Checkpoint status: **{overall}**.

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
| {int(single['captured'])} | {int(single['processed'])} | {int(single['dropped'])} | {int(single['queue_peak'])} | {single['fps']:.2f} | {single['p50_ms']:.2f} | {single['p90_ms']:.2f} | {single['p99_ms']:.2f} |

Capture timestamps flow through the queue and async result collection, so these are end-to-end
capture-to-result latencies rather than model-only timings.

Host RSS MiB start/peak/end: {memory['host_rss_mib']['start']:.2f} / {memory['host_rss_mib']['peak']:.2f} / {memory['host_rss_mib']['end']:.2f}.
Device memory MiB start/peak/end: {memory['device_memory_mib']['start']:.2f} / {memory['device_memory_mib']['peak']:.2f} / {memory['device_memory_mib']['end']:.2f}.

## Multi-Stream Fairness

Total throughput: {multi['total_fps']:.2f} frames/s.

| Stream | Captured | Processed | Dropped | FPS | P50 ms | P90 ms | P99 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{stream_rows}

Round-robin prevents a fast source from monopolizing every batch. Larger batches improve worker
efficiency but can increase timeout wait and tail latency; latest-first reduces stale work but may
drop more frames from bursty streams.

## CPU vs CUDA/NPP Preprocessing

| Memory mode | CPU ms | Host stage ms | H2D ms | GPU/NPP ms | D2H ms | Mean abs error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{cuda_table}

Mapped memory removes explicit transfers but makes a discrete GPU access host memory across PCIe;
the measured kernel time must be included before describing it as faster.

## Lifecycle, Faults, and Sanitizers

- Repeated start/stop: {evidence['restarts']['requested']} cycles, {evidence['restarts']['failures']} failures.
- Soak requested: {evidence['soak']['requested_minutes']:.3f} minutes across {evidence['soak']['cycles']} cycles, {evidence['soak']['failures']} failures.

| Fault | Return code | Expected nonzero |
| --- | ---: | --- |
{fault_rows}

| Gate | Status |
| --- | --- |
{gate_rows}

ThreadSanitizer output: `{evidence['sanitizers']['thread_sanitizer']['stderr'].strip() or 'no diagnostics'}`.
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
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path,
                        default=ROOT / "17a_pipeline_performance_report/outputs/evidence.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/17a_pipeline_performance.md")
    args = parser.parse_args()
    evidence = load_json(args.evidence)
    csv_path = ROOT / evidence["cuda_benchmark_csv"]
    with csv_path.open(newline="", encoding="utf-8") as handle:
        cuda_rows = list(csv.DictReader(handle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(evidence, cuda_rows), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
