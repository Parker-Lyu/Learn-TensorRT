#!/usr/bin/env python3
"""Summarize trtexec logs into a compact Markdown benchmark report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "06_trtexec_engine" / "outputs"
DEFAULT_REPORT = DEFAULT_OUTPUT_DIR / "benchmark_summary.md"


METRIC_PATTERNS = {
    "throughput_qps": re.compile(r"Throughput:\s*([0-9.]+)\s*qps"),
    "latency_ms": re.compile(r"Latency:\s*min\s*=\s*([0-9.]+)\s*ms,\s*max\s*=\s*([0-9.]+)\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms"),
    "enqueue_ms": re.compile(r"Enqueue Time:\s*min\s*=\s*([0-9.]+)\s*ms,\s*max\s*=\s*([0-9.]+)\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms"),
    "h2d_ms": re.compile(r"H2D Latency:\s*min\s*=\s*([0-9.]+)\s*ms,\s*max\s*=\s*([0-9.]+)\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms"),
    "gpu_compute_ms": re.compile(r"GPU Compute Time:\s*min\s*=\s*([0-9.]+)\s*ms,\s*max\s*=\s*([0-9.]+)\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms"),
    "d2h_ms": re.compile(r"D2H Latency:\s*min\s*=\s*([0-9.]+)\s*ms,\s*max\s*=\s*([0-9.]+)\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms"),
}

PERCENTILE_PATTERN = re.compile(r"Percentile:\s*([0-9.]+)%\s*=\s*([0-9.]+)\s*ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Markdown summary from lesson 06 trtexec outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory with lesson 06 outputs.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Markdown report path.")
    return parser.parse_args()


def load_manifest(output_dir: Path) -> dict[str, Any]:
    manifest_path = output_dir / "build_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run build_and_benchmark.py first.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def file_size_mib(path: Path) -> str:
    if not path.exists():
        return "missing"
    return f"{path.stat().st_size / (1024 * 1024):.2f}"


def extract_metrics(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {"status": "missing log"}

    text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics: dict[str, Any] = {"status": "ok" if "&&&& PASSED TensorRT.trtexec" in text else "check log"}

    throughput = METRIC_PATTERNS["throughput_qps"].search(text)
    if throughput:
        metrics["throughput_qps"] = float(throughput.group(1))

    for key in ("latency_ms", "enqueue_ms", "h2d_ms", "gpu_compute_ms", "d2h_ms"):
        match = METRIC_PATTERNS[key].search(text)
        if match:
            metrics[key] = {
                "min": float(match.group(1)),
                "max": float(match.group(2)),
                "mean": float(match.group(3)),
            }

    percentiles = {match.group(1): float(match.group(2)) for match in PERCENTILE_PATTERN.finditer(text)}
    if percentiles:
        metrics["percentiles_ms"] = percentiles

    return metrics


def metric_value(metrics: dict[str, Any], key: str, subkey: str | None = None) -> str:
    value = metrics.get(key)
    if value is None:
        return "-"
    if subkey is not None:
        if not isinstance(value, dict) or subkey not in value:
            return "-"
        value = value[subkey]
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def build_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# 06 trtexec Benchmark Summary",
        "",
        "Generated from `trtexec` logs and JSON artifacts.",
        "",
        "## Runtime Environment",
        "",
        f"- Course image: `{manifest['runtime_environment']['course_image']}`",
        f"- NVIDIA container release: `{manifest['runtime_environment']['nvidia_container_release']}`",
        f"- NVIDIA build ID: `{manifest['runtime_environment']['nvidia_build_id']}`",
        f"- TensorRT: `{manifest['runtime_environment']['tensorrt']}`",
        f"- CUDA Toolkit: `{manifest['runtime_environment']['cuda_toolkit']}`",
        f"- GPU and driver: `{manifest['runtime_environment']['gpu_and_driver']}`",
        f"- Python: `{manifest['runtime_environment']['python']}`",
        "",
        "## Build Settings",
        "",
        f"- Static ONNX: `{manifest['static_onnx']}`",
        f"- Dynamic ONNX: `{manifest['dynamic_onnx']}`",
        f"- Workspace: `{manifest['workspace_mib']} MiB`",
        f"- Warmup: `{manifest['warmup_ms']} ms`",
        f"- Duration: `{manifest['duration_sec']} s`",
        f"- Average runs: `{manifest['avg_runs']}`",
        "",
        "## Results",
        "",
        "| Engine | Precision | Dynamic | Engine MiB | Throughput qps | Latency mean ms | GPU compute mean ms | H2D mean ms | D2H mean ms | Status |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for build in manifest["builds"]:
        metrics = extract_metrics(Path(build["log"]))
        precision = "FP16" if build.get("fp16", build.get("precision") == "fp16") else "FP32"
        lines.append(
            "| {name} | {precision} | {dynamic} | {size} | {throughput} | {latency} | {gpu} | {h2d} | {d2h} | {status} |".format(
                name=build["name"],
                precision=precision,
                dynamic="yes" if build["dynamic"] else "no",
                size=file_size_mib(Path(build["engine"])),
                throughput=metric_value(metrics, "throughput_qps"),
                latency=metric_value(metrics, "latency_ms", "mean"),
                gpu=metric_value(metrics, "gpu_compute_ms", "mean"),
                h2d=metric_value(metrics, "h2d_ms", "mean"),
                d2h=metric_value(metrics, "d2h_ms", "mean"),
                status=metric_value(metrics, "status"),
            )
        )

    lines.extend(
        [
            "",
            "## Artifact Index",
            "",
            "| Engine | Log | Timing JSON | Layer Info JSON | Profile JSON |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for build in manifest["builds"]:
        lines.append(
            f"| `{build['engine']}` | `{build['log']}` | `{build['times']}` | "
            f"`{build['layers']}` | `{build['profile']}` |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Compare latency with GPU compute time before blaming TensorRT kernels; transfer and enqueue overheads matter.",
            "- FP16 should usually reduce engine size and improve GPU compute time on modern NVIDIA GPUs, but exact gains depend on hardware and tactics.",
            "- Dynamic engines need optimization profiles. Benchmark numbers are for the shape passed with `--shapes`, not for every shape in the profile.",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        output_dir = args.output_dir.resolve()
        manifest = load_manifest(output_dir)
        report = build_report(manifest)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
        print(f"report: {args.report.resolve()}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
