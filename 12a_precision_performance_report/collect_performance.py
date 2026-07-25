#!/usr/bin/env python3
"""Collect comparable per-inference trtexec samples for FP32, FP16, and INT8."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINES = {
    "fp32": ROOT / "12_yolov8_int8_quantization_engineering/outputs/tensorrt10/references/yolov8n_trt10_fp32.engine",
    "fp16": ROOT / "12_yolov8_int8_quantization_engineering/outputs/tensorrt10/references/yolov8n_trt10_fp16.engine",
    "int8": ROOT / "12_yolov8_int8_quantization_engineering/outputs/tensorrt10/candidate/yolov8n_qdq_int8.engine",
}
THROUGHPUT_PATTERN = re.compile(r"Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*qps")
EXPECTED_TRT_SERIES = "10.14"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(samples: list[dict]) -> dict:
    if len(samples) < 100:
        raise ValueError("at least 100 measured samples are required for P99")
    latency = [float(sample["latencyMs"]) for sample in samples]
    compute = [float(sample["computeMs"]) for sample in samples]
    return {
        "sample_count": len(samples),
        "latency_ms": {
            "mean": statistics.fmean(latency),
            "p50": percentile(latency, 0.50),
            "p90": percentile(latency, 0.90),
            "p99": percentile(latency, 0.99),
        },
        "gpu_compute_ms": {
            "mean": statistics.fmean(compute),
            "p50": percentile(compute, 0.50),
            "p90": percentile(compute, 0.90),
            "p99": percentile(compute, 0.99),
        },
    }


def parse_trtexec_throughput(output: str) -> float:
    """Read wall-time throughput reported by trtexec.

    Per-inference latency cannot be inverted to obtain throughput because trtexec may overlap
    transfers and compute from different inferences.
    """
    matches = THROUGHPUT_PATTERN.findall(output)
    if not matches:
        raise RuntimeError("could not parse throughput from trtexec output")
    throughput = float(matches[-1])
    if not math.isfinite(throughput) or throughput <= 0.0:
        raise RuntimeError(f"trtexec reported invalid throughput: {throughput}")
    return throughput


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return (result.stdout + result.stderr).strip()


def trtexec_version(executable: str) -> str:
    output = command_output([executable, "--version"])
    match = re.search(r"TensorRT v([0-9]+)", output)
    if not match:
        raise RuntimeError("could not parse TensorRT version from trtexec output")
    digits = match.group(1)
    if len(digits) == 4:
        major, minor, patch = digits[0], digits[1], digits[2:]
    elif len(digits) >= 6:
        major, minor, patch = digits[:-4], digits[-4:-2], digits[-2:]
    else:
        raise RuntimeError("unexpected compact TensorRT version")
    return f"{int(major)}.{int(minor)}.{int(patch)}"


def require_course_tensorrt(version: str) -> None:
    if not version.startswith(EXPECTED_TRT_SERIES + "."):
        raise RuntimeError(
            f"checkpoint 12a requires TensorRT {EXPECTED_TRT_SERIES}.x, found {version}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--warmup-ms", type=int, default=500)
    parser.add_argument("--trtexec", default=shutil.which("trtexec") or "/opt/tensorrt/bin/trtexec")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "12a_precision_performance_report/outputs/performance.json")
    args = parser.parse_args()
    if args.iterations < 100 or args.warmup_ms < 0:
        parser.error("iterations must be >=100 and warmup-ms must be non-negative")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    version = trtexec_version(args.trtexec)
    require_course_tensorrt(version)
    evidence = {
        "schema_version": 3,
        "methodology": {
            "tool": args.trtexec,
            "warmup_ms": args.warmup_ms,
            "iterations": args.iterations,
            "synchronization": "trtexec per-inference latency with H2D, compute, and D2H complete",
        },
        "environment": {
            "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version,pstate,power.limit",
                                   "--format=csv,noheader"]),
            "trtexec": version,
        },
        "backends": {},
    }
    for name, engine in DEFAULT_ENGINES.items():
        if not engine.is_file():
            raise FileNotFoundError(f"missing {name} engine: {engine}")
        times_path = args.output.parent / f"{name}_times.json"
        log_path = args.output.parent / f"{name}_trtexec.log"
        command = [args.trtexec, f"--loadEngine={engine}", f"--warmUp={args.warmup_ms}",
                   "--duration=0", f"--iterations={args.iterations}",
                   f"--exportTimes={times_path}"]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        trtexec_output = result.stdout + result.stderr
        log_path.write_text(trtexec_output, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name} trtexec failed; inspect {log_path}")
        samples = json.loads(times_path.read_text(encoding="utf-8"))
        evidence["backends"][name] = {
            "engine": str(engine.relative_to(ROOT)),
            "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
            "command": command,
            "throughput_qps": parse_trtexec_throughput(trtexec_output),
            **summarize(samples),
        }
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
