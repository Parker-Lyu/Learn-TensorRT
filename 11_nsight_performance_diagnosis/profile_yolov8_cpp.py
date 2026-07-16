#!/usr/bin/env python3
"""Measure steady-state latency and capture an Nsight Systems trace for lesson 10."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LATENCY_KEYS = [
    "preprocess",
    "h2d",
    "enqueue_host",
    "gpu_compute",
    "d2h",
    "postprocess",
    "total",
]
ADDITIVE_STAGE_KEYS = ["preprocess", "h2d", "gpu_compute", "d2h", "postprocess"]
COMPOSITION_KEYS = ["cpu", "transfer", "gpu_compute", "unaccounted"]


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
    if not 0.0 <= pct <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[index]


def summarize_values(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency summary requires at least one sample")
    return {
        "min": min(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def validate_latency_samples(raw_samples: Any, label: str) -> list[dict[str, float]]:
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError(f"{label} must be a non-empty list")
    samples: list[dict[str, float]] = []
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, dict):
            raise ValueError(f"{label}[{index}] must be an object")
        sample: dict[str, float] = {}
        for key in LATENCY_KEYS:
            value = raw_sample.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{label}[{index}].{key} must be numeric")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{label}[{index}].{key} must be finite and non-negative")
            sample[key] = numeric
        samples.append(sample)
    return samples


def summarize_latency(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {key: summarize_values([sample[key] for sample in samples]) for key in LATENCY_KEYS}


def composition_samples(samples: list[dict[str, float]]) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for sample in samples:
        total = sample["total"]
        if total <= 0.0:
            raise ValueError("total latency must be positive for composition analysis")
        accounted = sum(sample[key] for key in ADDITIVE_STAGE_KEYS)
        result.append(
            {
                "cpu": (sample["preprocess"] + sample["postprocess"]) / total,
                "transfer": (sample["h2d"] + sample["d2h"]) / total,
                "gpu_compute": sample["gpu_compute"] / total,
                "unaccounted": max(0.0, total - accounted) / total,
            }
        )
    return result


def summarize_composition(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    ratios = composition_samples(samples)
    return {key: summarize_values([sample[key] for sample in ratios]) for key in COMPOSITION_KEYS}


def classify_bottleneck(composition: dict[str, dict[str, float]]) -> dict[str, Any]:
    ranked = sorted(COMPOSITION_KEYS, key=lambda key: composition[key]["p50"], reverse=True)
    top, second = ranked[0], ranked[1]
    top_share = composition[top]["p50"]
    gap = top_share - composition[second]["p50"]
    dominant = top if top_share >= 0.50 and gap >= 0.15 else None
    descriptions = {
        "cpu": "CPU preprocessing and postprocessing dominate the typical measured request.",
        "transfer": "Host-device transfers dominate the typical measured request.",
        "gpu_compute": "TensorRT GPU compute dominates the typical measured request.",
        "unaccounted": "Unaccounted host/runtime overhead dominates the typical measured request.",
    }
    if dominant is None:
        diagnosis = "No category dominates strongly; use the Nsight timeline to inspect gaps and synchronization."
    else:
        diagnosis = descriptions[dominant]
    return {
        "method": "P50 per-request latency share; dominant requires >=50% share and >=15 percentage-point lead",
        "dominant_category": dominant,
        "top_category": top,
        "top_p50_share": top_share,
        "diagnosis": diagnosis,
    }


def build_lesson10(lesson10_dir: Path) -> None:
    run_command(["cmake", "-S", ".", "-B", "build"], cwd=lesson10_dir)
    run_command(["cmake", "--build", "build", "-j2"], cwd=lesson10_dir)


def application_command(
    args: argparse.Namespace,
    executable: Path,
    target_output_dir: Path,
    warmup_iterations: int,
    iterations: int,
) -> list[str]:
    return [
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
        "--warmup-iterations",
        str(warmup_iterations),
        "--iterations",
        str(iterations),
    ]


def load_trtexec_baseline(engine_path: Path) -> dict[str, Any]:
    times_path = engine_path.with_name(f"{engine_path.stem}_times.json")
    if not times_path.is_file():
        return {"available": False, "reason": f"timing JSON not found: {times_path}"}
    raw = json.loads(times_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"trtexec timing JSON must contain samples: {times_path}")
    mapping = {"h2d": "h2dMs", "gpu_compute": "computeMs", "d2h": "d2hMs", "total": "latencyMs"}
    summary: dict[str, dict[str, float]] = {}
    for output_key, input_key in mapping.items():
        values: list[float] = []
        for index, sample in enumerate(raw):
            value = sample.get(input_key) if isinstance(sample, dict) else None
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"invalid {input_key} at {times_path}[{index}]")
            values.append(float(value))
        summary[output_key] = summarize_values(values)
    return {"available": True, "path": str(times_path), "samples": len(raw), "latency_ms": summary}


def run_baseline(
    args: argparse.Namespace, executable: Path, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = output_dir / "baseline_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    command = application_command(args, executable, run_dir, args.warmup_iterations, args.iterations)
    command_result = run_command(command)
    report_path = run_dir / "detections.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"lesson 10 did not create its JSON report: {report_path}")
    application_report = json.loads(report_path.read_text(encoding="utf-8"))
    samples = validate_latency_samples(application_report.get("latency_samples_ms"), "latency_samples_ms")
    if len(samples) != args.iterations:
        raise ValueError(f"expected {args.iterations} measured samples, received {len(samples)}")

    raw_warmup = application_report.get("warmup_latency_samples_ms", [])
    warmup_samples = (
        validate_latency_samples(raw_warmup, "warmup_latency_samples_ms") if raw_warmup else []
    )
    if len(warmup_samples) != args.warmup_iterations:
        raise ValueError(f"expected {args.warmup_iterations} warmup samples, received {len(warmup_samples)}")

    composition = summarize_composition(samples)
    summary = {
        "warmup_iterations": args.warmup_iterations,
        "iterations": args.iterations,
        "latency_ms": summarize_latency(samples),
        "composition_ratio": composition,
        "heuristic_diagnosis": classify_bottleneck(composition),
        "first_warmup_latency_ms": warmup_samples[0] if warmup_samples else None,
        "command_elapsed_ms": command_result.elapsed_sec * 1000.0,
        "percentile_warning": (
            "P99 is descriptive only with fewer than 100 measured samples."
            if args.iterations < 100
            else ""
        ),
    }
    command_metadata = {
        "command": command,
        "returncode": command_result.returncode,
        "elapsed_sec": command_result.elapsed_sec,
        "stdout": command_result.stdout,
        "stderr": command_result.stderr,
        "report": str(report_path),
    }
    return application_report, {"summary": summary, "command": command_metadata}


def run_nsys(args: argparse.Namespace, executable: Path, output_dir: Path) -> dict[str, Any]:
    nsys = shutil.which("nsys")
    if nsys is None:
        return {"available": False, "reason": "nsys not found on PATH"}

    nsys_dir = output_dir / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)
    target_output_dir = nsys_dir / "target_run"
    target_output_dir.mkdir(parents=True, exist_ok=True)
    report_base = nsys_dir / "yolov8_trt_cpp"
    target_command = application_command(
        args, executable, target_output_dir, args.nsys_warmup_iterations, args.nsys_iterations
    )
    command = [
        nsys,
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--cuda-memory-usage=true",
        "--force-overwrite=true",
        "--output",
        str(report_base),
        *target_command,
    ]
    result = run_command(command, check=False)
    report_path = report_base.with_suffix(".nsys-rep")
    sqlite_path = report_base.with_suffix(".sqlite")
    export_result: dict[str, Any] | None = None
    stats_result: dict[str, Any] | None = None
    stats_path = nsys_dir / "nsys_stats.txt"
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
        export_result = command_result_dict(sqlite_export)

        stats_command = [
            nsys,
            "stats",
            "--force-export=true",
            "--report",
            "nvtx_pushpop_sum",
            "--report",
            "cuda_api_sum",
            "--report",
            "cuda_gpu_sum",
            str(report_path),
        ]
        stats = run_command(stats_command, check=False)
        stats_path.write_text(stats.stdout + stats.stderr, encoding="utf-8")
        stats_result = {
            "command": stats.command,
            "returncode": stats.returncode,
            "elapsed_sec": stats.elapsed_sec,
            "stderr": stats.stderr,
            "path": str(stats_path),
        }

    version = run_command([nsys, "--version"], check=False)
    return {
        "available": True,
        "version": (version.stdout + version.stderr).strip(),
        **command_result_dict(result),
        "report": str(report_path),
        "sqlite": str(sqlite_path) if sqlite_path.exists() else "",
        "sqlite_export": export_result,
        "stats": stats_result,
        "target_json": str(target_output_dir / "detections.json"),
    }


def command_result_dict(result: CommandResult) -> dict[str, Any]:
    return {
        "command": result.command,
        "returncode": result.returncode,
        "elapsed_sec": result.elapsed_sec,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def markdown_latency_table(summary: dict[str, dict[str, float]], keys: list[str]) -> list[str]:
    lines = [
        "| Stage | min | mean | P50 | P90 | P99 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in keys:
        stats = summary[key]
        lines.append(
            f"| {key} | {stats['min']:.3f} | {stats['mean']:.3f} | {stats['p50']:.3f} | "
            f"{stats['p90']:.3f} | {stats['p99']:.3f} | {stats['max']:.3f} |"
        )
    return lines


def write_markdown(
    path: Path,
    args: argparse.Namespace,
    baseline_summary: dict[str, Any],
    trtexec: dict[str, Any],
    nsys_report: dict[str, Any],
) -> None:
    latency = baseline_summary["latency_ms"]
    diagnosis = baseline_summary["heuristic_diagnosis"]
    lines = [
        "# Lesson 11 Diagnosis Report",
        "",
        f"- Engine: `{args.engine}`",
        f"- Image: `{args.image}`",
        f"- Warmup iterations: `{args.warmup_iterations}`",
        f"- Measured iterations: `{args.iterations}`",
        f"- Heuristic diagnosis: {diagnosis['diagnosis']}",
        f"- Method: {diagnosis['method']}",
    ]
    if baseline_summary["percentile_warning"]:
        lines.append(f"- Warning: {baseline_summary['percentile_warning']}")

    first_warmup = baseline_summary["first_warmup_latency_ms"]
    if first_warmup:
        lines.extend(
            [
                "",
                "## Cold First Inference",
                "",
                "The first warmup iteration is reported separately and excluded from steady-state percentiles.",
                "",
                f"- First pipeline total: `{first_warmup['total']:.3f} ms`",
                f"- First GPU compute: `{first_warmup['gpu_compute']:.3f} ms`",
            ]
        )

    lines.extend(["", "## Steady-State Latency (ms)", ""])
    lines.extend(markdown_latency_table(latency, LATENCY_KEYS))
    lines.extend(
        [
            "",
            "`enqueue_host` is CPU API-call time. `gpu_compute` is measured with CUDA events. They overlap",
            "conceptually and must not be added together when reconstructing total latency.",
            "",
            "## Typical Request Composition",
            "",
            "| Category | P50 share | P90 share |",
            "| --- | ---: | ---: |",
        ]
    )
    for key in COMPOSITION_KEYS:
        stats = baseline_summary["composition_ratio"][key]
        lines.append(f"| {key} | {stats['p50'] * 100.0:.1f}% | {stats['p90'] * 100.0:.1f}% |")

    lines.extend(["", "## trtexec Model-Only Reference", ""])
    if trtexec.get("available"):
        lines.append(f"Source: `{trtexec['path']}` ({trtexec['samples']} samples)")
        lines.append("")
        lines.extend(markdown_latency_table(trtexec["latency_ms"], ["h2d", "gpu_compute", "d2h", "total"]))
    else:
        lines.append(f"Unavailable: {trtexec.get('reason')}")

    lines.extend(["", "## Nsight Systems", ""])
    if nsys_report.get("available") and nsys_report.get("returncode") == 0:
        lines.extend(
            [
                f"- Version: `{nsys_report['version']}`",
                f"- Timeline: `{nsys_report['report']}`",
                f"- SQLite export: `{nsys_report['sqlite'] or 'not generated'}`",
                f"- Text statistics: `{(nsys_report.get('stats') or {}).get('path', 'not generated')}`",
                "",
                "Use the `warmup_iteration_*` and `measured_iteration_*` NVTX ranges to inspect:",
                "",
                "- whether the first inference differs from steady-state iterations",
                "- CPU gaps between preprocess, CUDA submission, synchronization, and postprocess",
                "- H2D, TensorRT kernels, and D2H ordering on the CUDA stream",
                "- blocking CUDA APIs or unexpectedly large unaccounted host overhead",
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

    lines.extend(
        [
            "",
            "## Measurement Boundaries",
            "",
            "- Steady-state totals include preprocessing, inference synchronization, and postprocessing.",
            "- Engine deserialization, image decoding, visualization, file writing, and process startup are excluded.",
            "- The automatic diagnosis is a latency-share heuristic; timeline evidence remains the final authority.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lesson10-dir", type=Path, default=Path("../10_yolov8_trt_cpp"))
    parser.add_argument("--engine", type=Path, default=Path("../06_trtexec_engine/outputs/yolov8n_static_fp32.engine"))
    parser.add_argument("--image", type=Path, default=Path("../assets/dog.webp"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--nsys-warmup-iterations", type=int, default=2)
    parser.add_argument("--nsys-iterations", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-nsys", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup_iterations < 0 or args.nsys_warmup_iterations < 0:
        raise ValueError("warmup iteration counts cannot be negative")
    if args.iterations <= 0 or args.nsys_iterations <= 0:
        raise ValueError("measured iteration counts must be positive")
    if not (0.0 <= args.confidence <= 1.0) or not (0.0 <= args.iou <= 1.0):
        raise ValueError("--confidence and --iou must be in [0, 1]")
    if args.max_detections <= 0:
        raise ValueError("--max-detections must be positive")
    for label, path in (("engine", args.engine), ("image", args.image)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_build:
        build_lesson10(args.lesson10_dir)

    executable = args.lesson10_dir / "build" / "yolov8_trt_cpp"
    if not executable.is_file():
        raise FileNotFoundError(f"lesson 10 executable not found: {executable}")

    application_report, baseline = run_baseline(args, executable, args.output_dir)
    trtexec = load_trtexec_baseline(args.engine)
    nsys_report: dict[str, Any] = {"available": False, "reason": "skipped by --skip-nsys"}
    if not args.skip_nsys:
        nsys_report = run_nsys(args, executable, args.output_dir)

    result = {
        "schema_version": 2,
        "artifacts": {
            "engine": {
                "path": str(args.engine),
                "sha256": hashlib.sha256(args.engine.read_bytes()).hexdigest(),
            },
            "image": {
                "path": str(args.image),
                "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
            },
        },
        "baseline_summary": baseline["summary"],
        "application_command": baseline["command"],
        "application_report": application_report,
        "trtexec_baseline": trtexec,
        "nsys": nsys_report,
    }
    summary_json = args.output_dir / "diagnosis_summary.json"
    summary_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary_md = args.output_dir / "diagnosis_report.md"
    write_markdown(summary_md, args, baseline["summary"], trtexec, nsys_report)

    diagnosis = baseline["summary"]["heuristic_diagnosis"]
    print(f"Warmup iterations: {args.warmup_iterations}")
    print(f"Measured iterations: {args.iterations}")
    print(f"Steady-state total P50: {baseline['summary']['latency_ms']['total']['p50']:.3f} ms")
    print(f"Heuristic diagnosis: {diagnosis['diagnosis']}")
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
