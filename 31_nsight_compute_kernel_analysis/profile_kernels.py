#!/usr/bin/env python3
"""Collect timing and profiler evidence for lesson 31 without conflating their roles."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LESSON = Path(__file__).resolve().parent
VARIANTS = ("unfused", "baseline_16x16", "block_32x8", "linear", "vectorized")
SECTIONS = (
    "LaunchStats",
    "Occupancy",
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "SchedulerStats",
    "WarpStateStats",
)
@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    elapsed_seconds: float
    stdout: str
    stderr: str


def run(command: list[str], *, check: bool = True, env: dict[str, str] | None = None) -> CommandResult:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=env,
    )
    result = CommandResult(
        command, completed.returncode, time.monotonic() - started,
        completed.stdout, completed.stderr,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return result


def tool_version(tool: str) -> str:
    executable = shutil.which(tool)
    if executable is None:
        return "unavailable"
    result = run([executable, "--version"], check=False)
    return (result.stdout + result.stderr).strip()


def environment_identity() -> dict[str, Any]:
    gpu = run(
        [
            "nvidia-smi",
            "--query-gpu=name,compute_cap,driver_version,memory.total,clocks.current.sm,"
            "clocks.current.memory,pstate,power.limit",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    return {
        "course_image": "nvcr.io/nvidia/pytorch:25.11-py3",
        "container_release": os.environ.get("NVIDIA_PYTORCH_VERSION", "unknown"),
        "container_build_id": os.environ.get("NVIDIA_BUILD_ID", "unknown"),
        "host": platform.platform(),
        "gpu_query": (gpu.stdout + gpu.stderr).strip(),
        "cuda_toolkit": os.environ.get("CUDA_VERSION", "unknown"),
        "nsight_systems": tool_version("nsys"),
        "nsight_compute": tool_version("ncu"),
    }


def benchmark_command(
    executable: Path, output: Path, warmup: int, iterations: int, variant: str = "all"
) -> list[str]:
    return [
        str(executable),
        "--variant", variant,
        "--warmup", str(warmup),
        "--iterations", str(iterations),
        "--output", str(output),
    ]


def nsys_command(
    nsys: str, lesson20_executable: Path, output_base: Path, iterations: int
) -> list[str]:
    return [
        nsys,
        "profile",
        "--force-overwrite=true",
        "--sample=none",
        "--trace=cuda,nvtx",
        "--output", str(output_base),
        str(lesson20_executable),
        "--iterations", str(iterations),
    ]


def ncu_command(
    ncu: str,
    executable: Path,
    output_base: Path,
    target_output: Path,
    variant: str,
    warmup: int,
) -> list[str]:
    launches = 2 if variant == "unfused" else 1
    command = [
        ncu,
        "--force-overwrite",
        "--export", str(output_base),
        "--replay-mode", "kernel",
        "--launch-skip", str(warmup * launches),
        "--launch-count", str(launches),
    ]
    for section in SECTIONS:
        command.extend(["--section", section])
    command.extend(benchmark_command(executable, target_output, warmup, 1, variant))
    return command


def parse_ncu_csv(text: str) -> list[dict[str, str]]:
    rows = list(csv.reader(io.StringIO(text)))
    header_index = next(
        (index for index, row in enumerate(rows) if "Kernel Name" in row),
        None,
    )
    if header_index is None:
        raise ValueError("Nsight Compute CSV does not contain a raw metric table")
    header = rows[header_index]
    if "Metric Name" not in header or "Metric Value" not in header:
        if header_index + 1 >= len(rows) or len(rows[header_index + 1]) != len(header):
            raise ValueError("Nsight Compute wide CSV does not contain a unit row")
        units = rows[header_index + 1]
        kernel_index = header.index("Kernel Name")
        metadata = {
            "ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Context", "Stream",
            "Block Size", "Grid Size", "Device", "CC",
        }
        parsed = []
        for row in rows[header_index + 2:]:
            if len(row) != len(header) or not row[kernel_index]:
                continue
            for index, name in enumerate(header):
                if name in metadata or not name or not row[index]:
                    continue
                parsed.append(
                    {
                        "Kernel Name": row[kernel_index],
                        "Metric Name": name,
                        "Metric Unit": units[index],
                        "Metric Value": row[index],
                    }
                )
        if not parsed:
            raise ValueError("Nsight Compute wide CSV metric table is empty")
        return parsed

    parsed: list[dict[str, str]] = []
    for row in rows[header_index + 1:]:
        if len(row) != len(header):
            continue
        item = dict(zip(header, row))
        if item.get("Metric Name"):
            parsed.append(item)
    if not parsed:
        raise ValueError("Nsight Compute CSV metric table is empty")
    return parsed


def summarize_ncu_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    summary = []
    for row in rows:
        name = row.get("Metric Name", "")
        keep = (
            name == "launch__registers_per_thread"
            or name == "sm__warps_active.avg.pct_of_peak_sustained_active"
            or name == "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
            or name in ("l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct")
            or name.startswith("smsp__average_warps_issue_stalled_")
        )
        if keep:
            summary.append(
                {
                    "kernel": row.get("Kernel Name", "unknown"),
                    "metric": name,
                    "unit": row.get("Metric Unit", ""),
                    "value": row.get("Metric Value", ""),
                }
            )
    if not summary:
        raise ValueError("Nsight Compute output did not contain the expected teaching metrics")
    return summary


def copy_lesson20_evidence(executable: Path, output_dir: Path, iterations: int) -> dict[str, Any]:
    if not executable.is_file():
        return {"available": False, "reason": f"lesson 20 executable not found: {executable}"}
    result = run([str(executable), "--iterations", str(iterations)])
    source_dir = executable.parent.parent / "outputs"
    copied = []
    for name in ("preprocess_benchmark.csv", "preprocess_benchmark_environment.json"):
        source = source_dir / name
        if not source.is_file():
            raise RuntimeError(f"lesson 20 did not produce expected evidence: {source}")
        destination = output_dir / f"lesson20_{name}"
        shutil.copy2(source, destination)
        copied.append(str(destination))
    return {"available": True, "command": asdict(result), "artifacts": copied}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--executable",
        type=Path,
        default=LESSON / "build/lesson31_kernel_benchmark",
    )
    parser.add_argument(
        "--lesson20-executable",
        type=Path,
        default=ROOT / "20_cuda_preprocess_npp/build/cuda_preprocess_npp",
    )
    parser.add_argument("--pipeline-metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, default=LESSON / "outputs")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--profile-warmup", type=int, default=5)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--skip-nsys", action="store_true")
    parser.add_argument("--skip-ncu", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0 or args.profile_warmup < 0:
        parser.error("warmup must be non-negative and iterations must be positive")
    executable = args.executable.resolve()
    if not executable.is_file():
        parser.error(f"benchmark executable does not exist: {executable}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "environment": environment_identity(),
        "measurement_policy": {
            "timing_source": "CUDA events in an unprofiled process",
            "profiler_duration_is_benchmark_evidence": False,
            "ncu_sections": list(SECTIONS),
        },
    }
    benchmark_output = output_dir / "kernel_benchmark.json"
    benchmark = run(benchmark_command(
        executable, benchmark_output, args.warmup, args.iterations
    ))
    manifest["kernel_benchmark"] = asdict(benchmark)
    manifest["lesson20"] = copy_lesson20_evidence(
        args.lesson20_executable.resolve(), output_dir, max(10, min(args.iterations, 50))
    )

    if args.pipeline_metrics is None:
        manifest["pipeline"] = {
            "available": False,
            "reason": "pass --pipeline-metrics from a matched lesson 21 run",
        }
    else:
        source = args.pipeline_metrics.resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output_dir / "lesson21_pipeline_metrics.json"
        shutil.copy2(source, destination)
        manifest["pipeline"] = {"available": True, "artifact": str(destination)}

    profiler_env = os.environ.copy()
    profiler_env.pop("LD_PRELOAD", None)
    if args.skip_nsys:
        manifest["nsight_systems"] = {"available": False, "reason": "skipped by request"}
    else:
        nsys = shutil.which("nsys")
        if nsys is None:
            raise RuntimeError("nsys is required unless --skip-nsys is used")
        nsys_dir = output_dir / "nsys"
        nsys_dir.mkdir(exist_ok=True)
        report_base = nsys_dir / "lesson20_preprocessing"
        lesson20_executable = args.lesson20_executable.resolve()
        if not lesson20_executable.is_file():
            raise RuntimeError("Lesson 20 executable is required for the Nsight Systems capture")
        result = run(nsys_command(
            nsys, lesson20_executable, report_base, max(3, args.profile_warmup)
        ), env=profiler_env)
        report = report_base.with_suffix(".nsys-rep")
        stats = run([
            nsys, "stats", "--force-export=true", "--report",
            "cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum",
            str(report),
        ], env=profiler_env)
        stats_path = nsys_dir / "nsys_stats.txt"
        stats_path.write_text(stats.stdout + stats.stderr, encoding="utf-8")
        manifest["nsight_systems"] = {
            "available": True,
            "capture": asdict(result),
            "report": str(report),
            "stats": str(stats_path),
        }

    metric_summaries: dict[str, Any] = {"schema_version": 1, "variants": {}}
    if args.skip_ncu:
        manifest["nsight_compute"] = {"available": False, "reason": "skipped by request"}
    else:
        ncu = shutil.which("ncu")
        if ncu is None:
            raise RuntimeError("ncu is required unless --skip-ncu is used")
        ncu_dir = output_dir / "ncu"
        ncu_dir.mkdir(exist_ok=True)
        captures = []
        for variant in args.variants:
            report_base = ncu_dir / variant
            result = run(ncu_command(
                ncu, executable, report_base, ncu_dir / f"{variant}_target.json",
                variant, args.profile_warmup,
            ), env=profiler_env)
            report = report_base.with_suffix(".ncu-rep")
            raw = run([ncu, "--import", str(report), "--page", "raw", "--csv"], env=profiler_env)
            rows = parse_ncu_csv(raw.stdout)
            metric_summaries["variants"][variant] = summarize_ncu_rows(rows)
            captures.append({"variant": variant, "capture": asdict(result), "report": str(report)})
        manifest["nsight_compute"] = {"available": True, "captures": captures}
    (output_dir / "ncu_metrics_summary.json").write_text(
        json.dumps(metric_summaries, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "profile_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "benchmark": str(benchmark_output),
        "manifest": str(output_dir / "profile_manifest.json"),
        "metrics": str(output_dir / "ncu_metrics_summary.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
