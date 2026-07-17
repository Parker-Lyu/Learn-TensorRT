#!/usr/bin/env python3
"""Collect matched trtexec throughput for FP16 and the ModelOpt Q/DQ engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LESSON_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = LESSON_DIR / "outputs/precision_recovery/05_modelopt_ptq/performance"
DEFAULT_ENGINES = {
    "fp16_reference": REPO_ROOT / "06_trtexec_engine/outputs/yolov8n_static_fp16.engine",
    "modelopt_qdq_int8_fp16": (
        LESSON_DIR
        / "outputs/precision_recovery/05_modelopt_ptq/"
        "yolov8n_modelopt_int8_max_train3000_trt86_int8_fp16.engine"
    ),
}
THROUGHPUT_PATTERN = re.compile(r"Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*qps")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
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
        raise ValueError("trtexec output contains no throughput")
    return float(matches[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--warmup-ms", type=int, default=500)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations < 100 or args.warmup_ms < 0:
        raise ValueError("iterations must be at least 100 and warmup must be non-negative")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": 1,
        "methodology": {
            "warmup_ms": args.warmup_ms,
            "iterations": args.iterations,
            "duration_sec": 0,
            "data_transfers": True,
            "throughput_source": "trtexec wall-time qps",
        },
        "engines": {},
    }
    for name, engine in DEFAULT_ENGINES.items():
        if not engine.is_file():
            raise FileNotFoundError(f"missing engine: {engine}")
        times_path = output_dir / f"{name}_times.json"
        log_path = output_dir / f"{name}_trtexec.log"
        command = [
            args.trtexec,
            f"--loadEngine={engine}",
            f"--warmUp={args.warmup_ms}",
            "--duration=0",
            f"--iterations={args.iterations}",
            f"--exportTimes={times_path}",
        ]
        result = subprocess.run(
            command, cwd=REPO_ROOT, text=True, capture_output=True, check=False
        )
        output = result.stdout + result.stderr
        log_path.write_text(output, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"trtexec failed for {name}; see {log_path}")
        samples = json.loads(times_path.read_text(encoding="utf-8"))
        report["engines"][name] = {
            "engine": str(engine.relative_to(REPO_ROOT)),
            "engine_sha256": sha256(engine),
            "command": command,
            "throughput_qps": parse_throughput(output),
            **summarize(samples),
        }
        print(f"Benchmarked {name}: {report['engines'][name]['throughput_qps']:.3f} qps")

    report_path = output_dir / "performance.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
