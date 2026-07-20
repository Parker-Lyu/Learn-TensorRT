#!/usr/bin/env python3
"""Collect matched TensorRT 10 FP32, FP16, and INT8 trtexec evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


LESSON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = LESSON_DIR.parent
STEP_OUTPUT = LESSON_DIR / "outputs/04_modelopt_qdq/trt10"
DEFAULT_ENGINES = {
    "fp32": STEP_OUTPUT / "references/yolov8n_trt10_fp32.engine",
    "fp16": STEP_OUTPUT / "references/yolov8n_trt10_fp16.engine",
    "int8": STEP_OUTPUT / "candidate/yolov8n_modelopt_hp_fp16_trt10.engine",
}
THROUGHPUT_PATTERN = re.compile(r"Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*qps")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile for empty samples")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(samples: list[dict]) -> dict:
    if len(samples) < 100:
        raise ValueError("at least 100 measured samples are required")

    def metric(name: str) -> dict[str, float]:
        values = [float(sample[name]) for sample in samples]
        return {
            "mean": statistics.fmean(values),
            "p50": percentile(values, 0.50),
            "p90": percentile(values, 0.90),
            "p99": percentile(values, 0.99),
        }

    return {
        "sample_count": len(samples),
        "latency_ms": metric("latencyMs"),
        "gpu_compute_ms": metric("computeMs"),
        "h2d_ms": metric("h2dMs"),
        "d2h_ms": metric("d2hMs"),
    }


def parse_throughput(text: str) -> float:
    matches = THROUGHPUT_PATTERN.findall(text)
    if not matches:
        raise ValueError("trtexec output contains no wall-time throughput")
    return float(matches[-1])


def benchmark_command(
    executable: str, engine: Path, times_path: Path, warmup_ms: int, iterations: int
) -> list[str]:
    return [
        executable,
        f"--loadEngine={engine}",
        f"--warmUp={warmup_ms}",
        "--duration=0",
        f"--iterations={iterations}",
        "--infStreams=1",
        f"--exportTimes={times_path}",
    ]


def gpu_state() -> dict[str, str]:
    fields = "name,driver_version,pstate,clocks.current.graphics,clocks.current.memory,power.draw"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        check=False,
    )
    return {"query": fields, "output": result.stdout.strip(), "error": result.stderr.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--warmup-ms", type=int, default=500)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--output-dir", type=Path, default=STEP_OUTPUT / "performance")
    parser.add_argument("--fp32-engine", type=Path, default=DEFAULT_ENGINES["fp32"])
    parser.add_argument("--fp16-engine", type=Path, default=DEFAULT_ENGINES["fp16"])
    parser.add_argument("--int8-engine", type=Path, default=DEFAULT_ENGINES["int8"])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations < 100 or args.warmup_ms < 0:
        raise ValueError("iterations must be at least 100 and warmup must be non-negative")
    engines = {"fp32": args.fp32_engine, "fp16": args.fp16_engine, "int8": args.int8_engine}
    missing = [str(path) for path in engines.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing engine(s): " + ", ".join(missing))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "warmup_ms": args.warmup_ms,
            "iterations": args.iterations,
            "duration_sec": 0,
            "data_transfers": True,
            "inference_streams": 1,
            "throughput_source": "trtexec wall-time qps",
        },
        "gpu_before": gpu_state(),
        "engines": {},
    }
    for name, engine in engines.items():
        engine = engine.resolve()
        times_path = output_dir / f"{name}_times.json"
        log_path = output_dir / f"{name}_trtexec.log"
        command = benchmark_command(
            args.trtexec, engine, times_path, args.warmup_ms, args.iterations
        )
        result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
        output = result.stdout + result.stderr
        log_path.write_text(output, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"trtexec failed for {name}; see {log_path}")
        samples = json.loads(times_path.read_text(encoding="utf-8"))
        report["engines"][name] = {
            "engine": str(engine),
            "engine_sha256": sha256(engine),
            "command": command,
            "log": str(log_path),
            "log_sha256": sha256(log_path),
            "times": str(times_path),
            "times_sha256": sha256(times_path),
            "throughput_qps": parse_throughput(output),
            **summarize(samples),
        }
        print(f"Benchmarked {name}: {report['engines'][name]['throughput_qps']:.3f} qps")
    report["gpu_after"] = gpu_state()
    report_path = output_dir / "performance.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Performance report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
