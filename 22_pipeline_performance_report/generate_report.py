#!/usr/bin/env python3
"""Generate the Lesson 22 report without turning absent evidence into a pass."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
PASS = "PASS"
FAIL = "FAIL"
INCOMPLETE = "INCOMPLETE"
NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class Gate:
    status: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        return {"status": self.status, "reason": self.reason}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evidence root must be an object")
    return value


def nested(value: Any, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def checked(description: str, predicate: Callable[[], bool]) -> Gate:
    try:
        passed = predicate()
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as error:
        return Gate(INCOMPLETE, f"malformed evidence: {error}")
    return Gate(PASS if passed else FAIL, description)


def available(description: str, predicate: Callable[[], bool], *required: Any) -> Gate:
    if any(value is None for value in required):
        return Gate(INCOMPLETE, f"missing evidence: {description}")
    return checked(description, predicate)


def run_passed(run: Any) -> bool:
    return isinstance(run, dict) and run.get("returncode") == 0


def accounting_passed(metrics: Any) -> bool:
    if not isinstance(metrics, dict):
        return False
    captured = int(metrics["captured"])
    terminal = sum(int(metrics.get(name, 0)) for name in
                   ("completed", "evicted", "failed", "aborted"))
    # Accept schema-v2 names only as an explicit migration aid; schema gate remains incomplete.
    if not any(name in metrics for name in ("completed", "evicted", "failed", "aborted")):
        terminal = int(metrics["processed"]) + int(metrics["dropped"])
    return captured == terminal


def evaluate(evidence: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Return required gates. Missing/malformed evidence is INCOMPLETE, never an exception."""
    gates: dict[str, Gate] = {}

    version = evidence.get("schema_version")
    gates["schema_v3"] = available(
        "schema_version must equal 3", lambda: int(version) == 3, version)

    platform = evidence.get("platform")
    gates["environment"] = available(
        "container, GPU, TensorRT, CUDA runtime/driver, and timestamp are required",
        lambda: all(nested(platform, key) not in (None, "") for key in
                    ("development_image", "gpu", "compute_capability", "driver",
                     "tensorrt", "cuda_runtime", "cuda_driver", "collected_at")),
        platform)

    batches = nested(evidence, "load_matrix", "batches")
    for size in (1, 2, 4):
        run = nested(batches, str(size))
        gates[f"real_batch_{size}"] = available(
            f"real Lesson 21 batch {size} run must succeed",
            lambda run=run, size=size: run_passed(run) and
            int(nested(run, "metrics", "batch_distribution", str(size), default=0)) > 0,
            run)

    overlap = nested(evidence, "load_matrix", "two_slot_overlap")
    gates["two_slot_overlap"] = available(
        "two distinct slots must be submitted before collection",
        lambda: run_passed(overlap) and bool(overlap["overlap_observed"]), overlap)

    references = evidence.get("reference_checks")
    for name in ("batch_vs_single", "cpu_vs_cuda_preprocess"):
        item = nested(references, name)
        gates[name] = available(
            f"{name} must pass its recorded tolerance",
            lambda item=item: run_passed(item) and bool(item["within_tolerance"]), item)

    multi = evidence.get("multi_stream")
    streams = nested(multi, "metrics", "streams")
    gates["real_multi_stream"] = available(
        "primary multi-stream evidence must come from Lesson 21 and contain two streams",
        lambda: run_passed(multi) and multi["producer"] == "lesson21" and
        isinstance(streams, list) and len(streams) >= 2 and
        all("stream_id" in stream and "p50_ms" in stream and "fps" in stream
            for stream in streams), multi, streams)

    policies = evidence.get("policies")
    for name in ("block", "drop_oldest", "latest_first"):
        item = nested(policies, name)
        gates[f"policy_{name}"] = available(
            f"{name} policy must succeed, remain bounded, and satisfy terminal accounting",
            lambda item=item: run_passed(item) and bool(item["bounded"]) and
            accounting_passed(item["metrics"]), item)

    faults = evidence.get("faults")
    required_faults = (
        "source_read", "invalid_shape", "insufficient_capacity", "tensor_address",
        "enqueue", "postprocess", "abort_pending")
    gates["integrated_fault_matrix"] = available(
        "all reproducible Lesson 21 fault hooks must return nonzero and clean up",
        lambda: all(nested(faults, name, "returncode") not in (None, 0) and
                    nested(faults, name, "cleanup_complete") is True
                    for name in required_faults), faults)

    restarts = evidence.get("restarts")
    if restarts is None:
        gates["restart_100"] = Gate(INCOMPLETE, "restart evidence is missing")
    else:
        try:
            requested_restarts = int(restarts["requested"])
        except (KeyError, TypeError, ValueError):
            gates["restart_100"] = Gate(INCOMPLETE, "restart evidence is malformed")
        else:
            gates["restart_100"] = (Gate(
                INCOMPLETE, "fewer than 100 restart cycles were requested")
                if requested_restarts < 100 else checked(
                    "at least 100 Lesson 21 processes must finish without failure",
                    lambda: int(restarts["failures"]) == 0))

    soak = evidence.get("long_lived_soak")
    if soak is None or not nested(soak, "formal_requested", default=False):
        gates["soak_30_minutes"] = Gate(
            INCOMPLETE, "a formal 30-minute single-process soak was not requested")
    else:
        gates["soak_30_minutes"] = available(
            "one Lesson 21 process must run for at least 30 minutes without failure",
            lambda: bool(soak["single_process"]) and float(soak["actual_seconds"]) >= 1800.0 and
            int(soak["failures"]) == 0 and run_passed(soak), soak)

    memory = evidence.get("memory_trend")
    for name in ("host", "device"):
        item = nested(memory, name)
        if memory is None or not nested(memory, "formal_requested", default=False):
            gates[f"{name}_memory_trend"] = Gate(
                INCOMPLETE, "formal long-lived memory sampling was not requested")
        elif item is None or not nested(item, "available", default=False):
            gates[f"{name}_memory_trend"] = Gate(
                INCOMPLETE, f"process-specific {name} memory samples are unavailable")
        else:
            gates[f"{name}_memory_trend"] = checked(
                f"{name} memory post-warmup window growth must remain within threshold",
                lambda item=item: int(item["sample_count"]) >= 2 and
                float(item["growth_percent"]) <= float(item["threshold_percent"]))

    sanitizers = evidence.get("sanitizers")
    for gate_name, evidence_name in (("lesson21_compute_memcheck", "compute_memcheck_lesson21"),
                                     ("lesson21_cpu_tsan", "lesson21_cpu_tsan")):
        item = nested(sanitizers, evidence_name)
        if item is None or not nested(item, "tool_started", default=False):
            gates[gate_name] = Gate(INCOMPLETE, f"{evidence_name} was not successfully started")
        else:
            gates[gate_name] = checked(
                f"{evidence_name} must exit successfully", lambda item=item: run_passed(item))

    return {name: gate.as_dict() for name, gate in gates.items()}


def overall_status(gates: dict[str, dict[str, str]]) -> str:
    statuses = [gate["status"] for gate in gates.values()]
    if FAIL in statuses:
        return FAIL
    if any(status != PASS for status in statuses):
        return INCOMPLETE
    return PASS


def display(value: Any, digits: int = 3) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def metric_rows(evidence: dict[str, Any]) -> tuple[str, str, str, str]:
    batch_rows = []
    batches = nested(evidence, "load_matrix", "batches", default={})
    if isinstance(batches, dict):
        for size in ("1", "2", "4"):
            metrics = nested(batches, size, "metrics", default={})
            batch_rows.append(
                f"| {size} | {display(nested(metrics, 'completed'))} | "
                f"{display(nested(metrics, 'fps'))} | {display(nested(metrics, 'p50_ms'))} | "
                f"{display(nested(metrics, 'p90_ms'))} | {display(nested(metrics, 'p99_ms'))} | "
                f"{display(nested(metrics, 'queue_peak'))} |")

    policy_rows = []
    policies = evidence.get("policies", {})
    if isinstance(policies, dict):
        for name in ("block", "drop_oldest", "latest_first"):
            metrics = nested(policies, name, "metrics", default={})
            policy_rows.append(
                f"| {name.replace('_', '-')} | {display(nested(metrics, 'captured'))} | "
                f"{display(nested(metrics, 'completed'))} | {display(nested(metrics, 'evicted'))} | "
                f"{display(nested(metrics, 'aborted'))} | {display(nested(metrics, 'queue_peak'))} | "
                f"{display(nested(metrics, 'fps'))} | {display(nested(metrics, 'p99_ms'))} |")

    stream_rows = []
    streams = nested(evidence, "multi_stream", "metrics", "streams", default=[])
    if isinstance(streams, list):
        for stream in streams:
            stream_rows.append(
                f"| {display(nested(stream, 'stream_id'))} | {display(nested(stream, 'completed'))} | "
                f"{display(nested(stream, 'fps'))} | {display(nested(stream, 'p50_ms'))} | "
                f"{display(nested(stream, 'p90_ms'))} | {display(nested(stream, 'p99_ms'))} |")

    fault_rows = []
    faults = evidence.get("faults", {})
    if isinstance(faults, dict):
        for name, item in sorted(faults.items()):
            fault_rows.append(
                f"| {name.replace('_', ' ')} | {display(nested(item, 'returncode'))} | "
                f"{display(nested(item, 'cleanup_complete'))} | {display(nested(item, 'timed_out'))} |")
    return ("\n".join(batch_rows) or "| no evidence | | | | | | |",
            "\n".join(policy_rows) or "| no evidence | | | | | | | |",
            "\n".join(stream_rows) or "| no evidence | | | | | |",
            "\n".join(fault_rows) or "| no evidence | | | |")


def render(evidence: dict[str, Any]) -> str:
    gates = evaluate(evidence)
    overall = overall_status(gates)
    rows = "\n".join(
        f"| {name.replace('_', ' ')} | {gate['status']} | {gate['reason']} |"
        for name, gate in gates.items())
    platform = evidence.get("platform") if isinstance(evidence.get("platform"), dict) else {}
    batch_rows, policy_rows, stream_rows, fault_rows = metric_rows(evidence)
    host_memory = nested(evidence, "memory_trend", "host", default={})
    device_memory = nested(evidence, "memory_trend", "device", default={})
    compute = nested(evidence, "sanitizers", "compute_memcheck_lesson21", default={})
    tsan = nested(evidence, "sanitizers", "lesson21_cpu_tsan", default={})
    return f"""# 22 - Pipeline Performance and Reliability Report

Generated from saved measurements. Checkpoint status: **{overall}**.

## Measurement Environment

- Development image: `{platform.get('development_image', 'not recorded')}`
- GPU: `{platform.get('gpu', platform.get('gpu_query', 'not recorded'))}`
- TensorRT: `{platform.get('tensorrt', platform.get('tensorrt_version', 'not recorded'))}`
- CUDA runtime / driver: `{platform.get('cuda_runtime', 'not recorded')}` / `{platform.get('cuda_driver', 'not recorded')}`
- Collected at: `{platform.get('collected_at', 'not recorded')}`

Performance values are valid only for the recorded environment. A missing tool, duration, field, or
hardware result is `INCOMPLETE`; it is never silently treated as a pass.

## Real Integrated Load Matrix

All primary rows below come from the Lesson 21 TensorRT executable, not simulated worker delays.
Latency is capture-to-result and uses the host monotonic clock.

| Batch | Completed | FPS | P50 ms | P90 ms | P99 ms | Queue peak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{batch_rows}

## Real Multi-Stream Result

Primary producer: `{nested(evidence, 'multi_stream', 'producer', default='missing')}`.

| Stream | Completed | FPS | P50 ms | P90 ms | P99 ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
{stream_rows}

## Overload and Freshness Policies

| Policy | Captured | Completed | Evicted | Aborted | Queue peak | FPS | P99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{policy_rows}

`block` preserves accepted work at the cost of producer backpressure. `drop-oldest` bounds stale
work by making eviction explicit. `latest-first` additionally discards superseded queued frames;
its freshness gain must be interpreted together with the evicted count and tail latency.

## Reference Checks

- Batch versus single-image return code / within tolerance:
  `{nested(evidence, 'reference_checks', 'batch_vs_single', 'returncode', default='missing')}` /
  `{nested(evidence, 'reference_checks', 'batch_vs_single', 'within_tolerance', default='missing')}`
- CPU versus CUDA preprocessing return code / within tolerance:
  `{nested(evidence, 'reference_checks', 'cpu_vs_cuda_preprocess', 'returncode', default='missing')}` /
  `{nested(evidence, 'reference_checks', 'cpu_vs_cuda_preprocess', 'within_tolerance', default='missing')}`

## Reliability and Memory

- Restart requested / failures: `{nested(evidence, 'restarts', 'requested', default='missing')}` /
  `{nested(evidence, 'restarts', 'failures', default='missing')}`
- Long-lived single process / actual seconds / failures:
  `{nested(evidence, 'long_lived_soak', 'single_process', default='missing')}` /
  `{display(nested(evidence, 'long_lived_soak', 'actual_seconds'))}` /
  `{nested(evidence, 'long_lived_soak', 'failures', default='missing')}`
- Host memory samples / growth / threshold percent:
  `{nested(host_memory, 'sample_count', default='missing')}` /
  `{display(nested(host_memory, 'growth_percent'))}` /
  `{display(nested(host_memory, 'threshold_percent'))}`
- Device memory samples / growth / threshold percent:
  `{nested(device_memory, 'sample_count', default='missing')}` /
  `{display(nested(device_memory, 'growth_percent'))}` /
  `{display(nested(device_memory, 'threshold_percent'))}`

Memory samples are process-specific. When Docker and NVIDIA tools expose different PID namespaces,
the collector accepts only one newly appearing GPU process and never sums unrelated compute apps.

## Integrated Fault Matrix

| Fault | Return code | Cleanup complete | Timed out |
| --- | ---: | --- | --- |
{fault_rows}

## Sanitizers

- Lesson 21 compute-sanitizer started / return code:
  `{nested(compute, 'tool_started', default='missing')}` /
  `{nested(compute, 'returncode', default='missing')}`
- Lesson 21 CPU TSAN started / return code:
  `{nested(tsan, 'tool_started', default='missing')}` /
  `{nested(tsan, 'returncode', default='missing')}`

## Acceptance Gates

| Gate | Status | Evidence requirement |
| --- | --- | --- |
{rows}

## Evidence Summary

- Schema version: `{evidence.get('schema_version', 'missing')}`
- Restart cycles: `{nested(evidence, 'restarts', 'requested', default='missing')}`
- Long-lived soak seconds: `{nested(evidence, 'long_lived_soak', 'actual_seconds', default='missing')}`
- Primary multi-stream producer: `{nested(evidence, 'multi_stream', 'producer', default='missing')}`

## Reproduction

```bash
python3 22_pipeline_performance_report/collect_pipeline_evidence.py
python3 22_pipeline_performance_report/generate_report.py
```

Formal evidence:

```bash
python3 22_pipeline_performance_report/collect_pipeline_evidence.py \\
  --soak-minutes 30 --restart-cycles 100
python3 22_pipeline_performance_report/generate_report.py
```
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path,
                        default=ROOT / "22_pipeline_performance_report/outputs/evidence.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/22_pipeline_performance.md")
    args = parser.parse_args()
    try:
        evidence = load_json(args.evidence)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        evidence = {"schema_version": None, "load_error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(evidence), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
