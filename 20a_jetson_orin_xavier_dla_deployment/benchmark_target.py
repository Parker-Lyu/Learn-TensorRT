#!/usr/bin/env python3
"""Benchmark target-built GPU and DLA engines from individual trtexec samples."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(samples: list[dict]) -> dict:
    if len(samples) < 100:
        raise ValueError("at least 100 samples are required")
    latency = [float(sample["latencyMs"]) for sample in samples]
    compute = [float(sample["computeMs"]) for sample in samples]
    return {"samples": len(samples),
            "latency_ms": {"mean": statistics.fmean(latency),
                           "p50": percentile(latency, 0.50),
                           "p90": percentile(latency, 0.90),
                           "p99": percentile(latency, 0.99)},
            "compute_ms": {"mean": statistics.fmean(compute),
                           "p50": percentile(compute, 0.50)},
            "throughput_images_per_second": 1000.0 / statistics.fmean(latency)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--warmup-ms", type=int, default=1000)
    args = parser.parse_args()
    if args.iterations < 100:
        parser.error("iterations must be >=100")
    output = ROOT / "outputs"
    engines = {"gpu": output / "yolov8n_jetson_gpu_fp16.engine",
               "dla_with_gpu_fallback": output / "yolov8n_jetson_dla_fp16.engine"}
    manifest = output / "platform_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("run check_platform.py on the target before benchmarking")
    evidence = {"methodology": {"iterations": args.iterations, "warmup_ms": args.warmup_ms},
                "platform": json.loads(manifest.read_text(encoding="utf-8")),
                "backends": {}}
    for name, engine in engines.items():
        if not engine.is_file():
            raise FileNotFoundError(f"missing target-built engine: {engine}")
        times = output / f"{name}_times.json"
        command = ["trtexec", f"--loadEngine={engine}", f"--warmUp={args.warmup_ms}",
                   "--duration=0", f"--iterations={args.iterations}", f"--exportTimes={times}"]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        (output / f"{name}_benchmark.log").write_text(
            result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name} benchmark failed")
        evidence["backends"][name] = summarize(json.loads(times.read_text(encoding="utf-8")))
    (output / "benchmark_summary.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
