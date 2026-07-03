#!/usr/bin/env python3
"""Run latency and Nsight Systems profiling for the lesson 10 C++ YOLOv8 app."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LATENCY_KEYS = ["preprocess", "h2d", "enqueue", "d2h", "postprocess", "total"]


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    elapsed_sec: float
    stdout: str
    stderr: str


def run_command(command: list[str], cwd: Path | None = None, check: bool = True) -> CommandResult:
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    elapsed = time.perf_counter() - start
    result = CommandResult(command, completed.returncode, elapsed, completed.stdout, completed.stderr)
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return result


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[index]


def summarize_latency(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for key in LATENCY_KEYS:
        values = [sample[key] for sample in samples]
        summary[key] = {
            "min": min(values),
            "mean": sum(values) / len(values),
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p99": percentile(values, 99),
            "max": max(values),
        }
    return summary


def dominant_stage(summary: dict[str, dict[str, float]]) -> str:
    candidates = ["preprocess", "h2d", "enqueue", "d2h", "postprocess"]
    return max(candidates, key=lambda key: summary[key]["p50"])


def classify_bottleneck(summary: dict[str, dict[str, float]]) -> str:
    stage = dominant_stage(summary)
    gpu_ms = summary["h2d"]["p50"] + summary["enqueue"]["p50"] + summary["d2h"]["p50"]
    cpu_ms = summary["preprocess"]["p50"] + summary["postprocess"]["p50"]
    if stage == "enqueue":
        return "GPU compute is the largest measured stage; inspect TensorRT kernels and precision."
    if cpu_ms > gpu_ms:
        return "CPU-side preprocessing/postprocessing dominates the median path; the GPU may be waiting between requests."
    if stage in {"h2d", "d2h"}:
        return "Host-device transfer is prominent; inspect pinned memory use, tensor size, and copy placement."
    return "No single stage dominates strongly; inspect the Nsight Systems timeline for gaps and synchronization."


def build_lesson10(lesson10_dir: Path) -> None:
    run_command(["cmake", "-S", ".", "-B", "build"], cwd=lesson10_dir)
    run_command(["cmake", "--build", "build", "-j2"], cwd=lesson10_dir)


def run_baseline(args: argparse.Namespace, executable: Path, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runs_dir = output_dir / "baseline_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_reports: list[dict[str, Any]] = []
    latency_samples: list[dict[str, float]] = []
    for index in range(args.iterations):
        run_dir = runs_dir / f"run_{index:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(executable),
            "--engine",
            str(args.engine),
            "--image",
            str(args.image),
            "--output-dir",
            str(run_dir),
            "--confidence",
            str(args.confidence),
            "--iou",
            str(args.iou),
            "--max-detections",
            str(args.max_detections),
        ]
        result = run_command(command)
        report_path = run_dir / "detections.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["command_elapsed_ms"] = result.elapsed_sec * 1000.0
        report["stdout"] = result.stdout
        report["stderr"] = result.stderr
        run_reports.append(report)
        latency_samples.append(report["latency_ms"])

    summary = {
        "iterations": args.iterations,
        "latency_ms": summarize_latency(latency_samples),
    }
    summary["dominant_stage_p50"] = dominant_stage(summary["latency_ms"])
    summary["diagnosis"] = classify_bottleneck(summary["latency_ms"])
    return run_reports, summary


def run_nsys(args: argparse.Namespace, executable: Path, output_dir: Path) -> dict[str, Any]:
    nsys = shutil.which("nsys")
    if nsys is None:
        return {"available": False, "reason": "nsys not found on PATH"}

    nsys_dir = output_dir / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)
    target_output_dir = nsys_dir / "target_run"
    target_output_dir.mkdir(parents=True, exist_ok=True)
    report_base = nsys_dir / "yolov8_trt_cpp"
    command = [
        nsys,
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--cuda-memory-usage=true",
        "--force-overwrite=true",
        "--output",
        str(report_base),
        str(executable),
        "--engine",
        str(args.engine),
        "--image",
        str(args.image),
        "--output-dir",
        str(target_output_dir),
        "--confidence",
        str(args.confidence),
        "--iou",
        str(args.iou),
        "--max-detections",
        str(args.max_detections),
    ]
    result = run_command(command, check=False)
    report_path = report_base.with_suffix(".nsys-rep")
    sqlite_path = report_base.with_suffix(".sqlite")
    export_result: dict[str, Any] | None = None
    if result.returncode == 0 and report_path.exists():
        export_command = [
            nsys,
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "--output",
            str(sqlite_path),
            str(report_path),
        ]
        sqlite_export = run_command(export_command, check=False)
        export_result = {
            "command": export_command,
            "returncode": sqlite_export.returncode,
            "elapsed_sec": sqlite_export.elapsed_sec,
            "stdout": sqlite_export.stdout,
            "stderr": sqlite_export.stderr,
        }
    return {
        "available": True,
        "command": command,
        "returncode": result.returncode,
        "elapsed_sec": result.elapsed_sec,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "report": str(report_path),
        "sqlite": str(sqlite_path) if sqlite_path.exists() else "",
        "sqlite_export": export_result,
        "target_json": str(target_output_dir / "detections.json"),
    }


def write_markdown(path: Path, args: argparse.Namespace, baseline_summary: dict[str, Any], nsys_report: dict[str, Any]) -> None:
    latency = baseline_summary["latency_ms"]
    lines = [
        "# Lesson 11 Diagnosis Report",
        "",
        f"- Engine: `{args.engine}`",
        f"- Image: `{args.image}`",
        f"- Iterations: `{args.iterations}`",
        f"- Dominant P50 stage: `{baseline_summary['dominant_stage_p50']}`",
        f"- Diagnosis: {baseline_summary['diagnosis']}",
        "",
        "## Latency Summary",
        "",
        "| Stage | min | mean | P50 | P90 | P99 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in LATENCY_KEYS:
        stats = latency[key]
        lines.append(
            f"| {key} | {stats['min']:.3f} | {stats['mean']:.3f} | {stats['p50']:.3f} | "
            f"{stats['p90']:.3f} | {stats['p99']:.3f} | {stats['max']:.3f} |"
        )

    lines.extend(["", "## Nsight Systems", ""])
    if nsys_report.get("available") and nsys_report.get("returncode") == 0:
        lines.extend(
            [
                f"- Report: `{nsys_report['report']}`",
                f"- SQLite export: `{nsys_report['sqlite'] or 'not generated'}`",
                "",
                "Open the `.nsys-rep` file with Nsight Systems and inspect:",
                "",
                "- whether CUDA kernels are packed together or separated by CPU gaps",
                "- whether H2D, TensorRT enqueue, and D2H happen on one stream in order",
                "- whether GPU work is short compared with CPU preprocessing/postprocessing",
                "- whether unexpected synchronizations appear before or after inference",
            ]
        )
    elif nsys_report.get("available"):
        lines.extend(
            [
                "Nsight Systems was found, but capture failed.",
                "",
                f"- Return code: `{nsys_report.get('returncode')}`",
                f"- stderr: `{nsys_report.get('stderr', '').strip()}`",
            ]
        )
    else:
        lines.append(f"Nsight Systems capture was skipped: {nsys_report.get('reason')}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson10-dir", type=Path, default=Path("../10_yolov8_trt_cpp"))
    parser.add_argument("--engine", type=Path, default=Path("../06_trtexec_engine/outputs/yolov8n_static_fp32.engine"))
    parser.add_argument("--image", type=Path, default=Path("../assets/dog.webp"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-nsys", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if not (0.0 <= args.confidence <= 1.0) or not (0.0 <= args.iou <= 1.0):
        raise ValueError("--confidence and --iou must be in [0, 1]")
    if args.max_detections <= 0:
        raise ValueError("--max-detections must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        build_lesson10(args.lesson10_dir)

    executable = args.lesson10_dir / "build" / "yolov8_trt_cpp"
    if not executable.is_file():
        raise FileNotFoundError(f"lesson 10 executable not found: {executable}")

    run_reports, baseline_summary = run_baseline(args, executable, args.output_dir)
    nsys_report = {"available": False, "reason": "skipped by --skip-nsys"}
    if not args.skip_nsys:
        nsys_report = run_nsys(args, executable, args.output_dir)

    result = {
        "baseline_summary": baseline_summary,
        "baseline_runs": run_reports,
        "nsys": nsys_report,
    }
    summary_json = args.output_dir / "diagnosis_summary.json"
    summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_md = args.output_dir / "diagnosis_report.md"
    write_markdown(summary_md, args, baseline_summary, nsys_report)

    print(f"Baseline iterations: {args.iterations}")
    print(f"Dominant P50 stage: {baseline_summary['dominant_stage_p50']}")
    print(f"Diagnosis: {baseline_summary['diagnosis']}")
    print(f"Summary JSON: {summary_json}")
    print(f"Report Markdown: {summary_md}")
    if nsys_report.get("available") and nsys_report.get("returncode") == 0:
        print(f"Nsight report: {nsys_report['report']}")
    elif nsys_report.get("available"):
        print("Nsight capture failed; inspect diagnosis_summary.json for stderr.")
    else:
        print(f"Nsight capture skipped: {nsys_report.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
