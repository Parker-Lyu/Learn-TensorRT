#!/usr/bin/env python3
"""Summarize verified trtexec evidence, keeping latency and compute percentiles separate."""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "06_trtexec_engine/outputs"
NUMBER = r"[0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?"
LABELS = {"latency_ms": "Latency", "enqueue_ms": "Enqueue Time", "h2d_ms": "H2D Latency",
          "gpu_compute_ms": "GPU Compute Time", "d2h_ms": "D2H Latency"}
PERCENTILE_PATTERN = re.compile(rf"percentile\(({NUMBER})%\)\s*=\s*({NUMBER})\s*ms", re.I)
CONTEXT_MEMORY_PATTERN = re.compile(
    rf"TensorRT-managed allocation in IExecutionContext creation:.*now:\s*CPU\s+({NUMBER}),\s*GPU\s+({NUMBER})\s*\(MiB\)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, help="Defaults to <output-dir>/benchmark_summary.md")
    return parser.parse_args(argv)


def load_manifest(output_dir):
    path = output_dir / "build_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}; run build_and_benchmark.py first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("builds"):
        raise ValueError("Manifest contains no completed benchmarks")
    return manifest


def extract_metrics(log_path: Path) -> dict[str, Any]:
    if not log_path.is_file():
        return {"status": "missing log"}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics = {"status": "ok" if "&&&& PASSED TensorRT.trtexec" in text else "failed or incomplete log"}
    throughput = re.search(rf"\bThroughput:\s*({NUMBER})\s*qps", text)
    if throughput:
        metrics["throughput_qps"] = float(throughput.group(1))
    for key, label in LABELS.items():
        # Anchor the label after [I], so Latency never matches H2D/D2H Latency.
        match = re.search(rf"\[I\]\s+{re.escape(label)}:\s*([^\n]+)", text)
        if not match:
            continue
        line = match.group(1)
        values = {}
        for statistic in ("min", "max", "mean"):
            value = re.search(rf"\b{statistic}\s*=\s*({NUMBER})\s*ms", line)
            if value:
                values[statistic] = float(value.group(1))
        values["percentiles"] = {str(float(p)).removesuffix(".0"): float(v)
                                 for p, v in PERCENTILE_PATTERN.findall(line)}
        metrics[key] = values
    memory = CONTEXT_MEMORY_PATTERN.findall(text)
    if memory:
        metrics["context_cpu_mib"], metrics["context_gpu_mib"] = map(float, memory[-1])
    return metrics


def metric_value(metrics, key, subkey=None):
    value = metrics.get(key)
    if subkey is not None:
        value = value.get(subkey) if isinstance(value, dict) else None
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def verified_metrics(build):
    metrics = extract_metrics(Path(build["log"]))
    if metrics["status"] != "ok":
        raise ValueError(f"{build['name']}: {metrics['status']}")
    for key in ("engine", "times", "layers", "profile"):
        path = Path(build[key])
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty {key} evidence: {path}")
        if key != "engine":
            payload = json.loads(path.read_text(encoding="utf-8"))
            valid = (isinstance(payload, dict) and bool(payload.get("Layers"))) if key == "layers" else (
                isinstance(payload, list) and bool(payload))
            if not valid:
                raise ValueError(f"JSON export contains no evidence: {path}")
    if "throughput_qps" not in metrics:
        raise ValueError(f"{build['name']}: missing throughput")
    for key in LABELS:
        values = metrics.get(key, {})
        if "mean" not in values or not {"50", "90", "95", "99"} <= values.get("percentiles", {}).keys():
            raise ValueError(f"{build['name']}: missing statistics for {key}")
    return metrics


def build_report(manifest):
    # Validate all evidence before writing any report, rather than silently publishing '-' results.
    rows = [(build, verified_metrics(build)) for build in manifest["builds"]]
    environment = manifest["runtime_environment"]
    lines = ["# 06 trtexec Benchmark Summary", "", "## Runtime Environment", ""]
    for key, value in environment.items():
        value = str(value).replace("\n", "; ")
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Measurement Conditions", "",
                  f"- Workspace limit: {manifest['workspace_mib']} MiB (builder workspace, not total GPU memory).",
                  f"- Warmup: {manifest['warmup_ms']} ms; duration: {manifest['duration_sec']} s; avgRuns: {manifest['avg_runs']}.",
                  f"- Timing cache: `{manifest.get('timing_cache', 'not recorded')}`.",
                  "- All runs include H2D/D2H transfers, one inference stream, random trtexec inputs, and a separate profiling run.",
                  "- FP32 disables TF32; FP16 keeps TensorRT's default TF32 policy for remaining FP32 work. This is a deployment-mode comparison, not an isolated FP16-only speedup.",
                  "- These measurements exclude application image preprocessing, decode, and NMS; qps counts inference batches.",
                  "", "| Engine | Typing | TF32 policy | Runtime shape | Min | Opt | Max | Reused engine | Measured UTC |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"])
    for build, _ in rows:
        shape = build.get("shapes") or json.dumps(build.get("onnx_input_shapes", "unrecorded static shape"))
        lines.append(f"| {build['name']} | {build.get('typing_mode', 'unrecorded')} | {build.get('tf32_policy', 'unrecorded')} | "
                     f"{shape} | {build.get('min_shapes') or '-'} | {build.get('opt_shapes') or '-'} | "
                     f"{build.get('max_shapes') or '-'} | {build.get('reused_engine', 'unrecorded')} | {build.get('measured_at_utc', 'unrecorded')} |")
    lines.extend(["", "## Results", "",
                  "| Engine | Precision | Engine file MiB | Throughput qps | Latency mean ms | GPU compute mean ms | H2D mean ms | D2H mean ms | TRT context CPU MiB | TRT context GPU MiB |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for build, metrics in rows:
        size = Path(build["engine"]).stat().st_size / (1024 * 1024)
        values = [metric_value(metrics, "throughput_qps")]
        values += [metric_value(metrics, key, "mean") for key in ("latency_ms", "gpu_compute_ms", "h2d_ms", "d2h_ms")]
        values += [metric_value(metrics, key) for key in ("context_cpu_mib", "context_gpu_mib")]
        lines.append(f"| {build['name']} | {build['precision']} | {size:.2f} | " + " | ".join(values) + " |")
    lines.extend(["", "Context memory is TensorRT-managed memory after context creation, as reported by trtexec. It is not total process memory or peak GPU memory; '-' means the runtime did not report it. Engine file size is disk storage.",
                  "", "## Tail Latency", "",
                  "| Engine | Metric | P50 ms | P90 ms | P95 ms | P99 ms |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"])
    for build, metrics in rows:
        for key in ("latency_ms", "gpu_compute_ms"):
            values = metrics[key]["percentiles"]
            lines.append(f"| {build['name']} | {LABELS[key]} | " +
                         " | ".join(f"{values[p]:.3f}" for p in ("50", "90", "95", "99")) + " |")
    lines.extend(["", "## Artifact Index", "",
                  "| Engine | Log | Timing JSON | Layer Info JSON | Profile JSON | Conversion gate |",
                  "| --- | --- | --- | --- | --- | --- |"])
    for build, _ in rows:
        values = [str(build.get(key) or "-") for key in ("engine", "log", "times", "layers", "profile", "validation_report")]
        lines.append("| " + " | ".join(f"`{value}`" for value in values) + " |")
    lines.extend(["", "## Commands", ""])
    for build, _ in rows:
        lines.extend([f"### {build['name']}", "", "```bash", shlex.join(build.get("command", [])) or "Command not recorded in this older manifest.", "```", ""])
    lines.extend(["## Interpretation", "",
                  "- Compare static FP32 with static FP16 at the same input shape and measurement settings. Dynamic results apply only to the measured shape, not every point in the profile.",
                  "- Latency includes H2D, GPU compute, and D2H. Enqueue and pipelining mean throughput is not generally 1000 / latency.",
                  "- Layer profiles come from a separate run; inspect the profile and layer-info JSON to explain precision choices and expensive layers.",
                  "- Reusing an engine still performs a fresh benchmark. Cache hits are not accuracy evidence; the conversion gate is a single-input raw-output check, not mAP validation."])
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    try:
        output_dir = args.output_dir.resolve()
        report = build_report(load_manifest(output_dir))
        report_path = (args.report or output_dir / "benchmark_summary.md").resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        print(f"report: {report_path}")
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
