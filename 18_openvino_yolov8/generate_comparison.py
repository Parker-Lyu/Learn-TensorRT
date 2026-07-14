#!/usr/bin/env python3
"""Compare OpenVINO CPU evidence with lesson 12a TensorRT GPU evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvino", type=Path,
                        default=ROOT / "18_openvino_yolov8/outputs/openvino_benchmark.json")
    parser.add_argument("--tensorrt", type=Path,
                        default=ROOT / "12a_precision_performance_report/outputs/performance.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "18_openvino_yolov8/outputs/comparison.md")
    args = parser.parse_args()
    ov = json.loads(args.openvino.read_text(encoding="utf-8"))
    trt = json.loads(args.tensorrt.read_text(encoding="utf-8"))
    rows = [
        ("OpenVINO CPU sync", ov["sync"]["latency_ms"], ov["sync"]["throughput_requests_per_second"]),
        ("OpenVINO CPU async", ov["async"]["latency_ms"], ov["async"]["throughput_requests_per_second"]),
    ]
    for precision in ("fp32", "fp16", "int8"):
        backend = trt["backends"][precision]
        rows.append((f"TensorRT GPU {precision.upper()}", backend["latency_ms"],
                     backend["throughput_images_per_second"]))
    table = "\n".join(
        f"| {name} | {latency['mean']:.3f} | {latency['p50']:.3f} | "
        f"{latency['p90']:.3f} | {latency['p99']:.3f} | {throughput:.1f} |"
        for name, latency, throughput in rows)
    text = f"""# OpenVINO CPU vs TensorRT GPU

| Backend | Mean ms | P50 ms | P90 ms | P99 ms | Requests/s |
| --- | ---: | ---: | ---: | ---: | ---: |
{table}

These are different devices and execution modes, so the table is a deployment comparison rather
than a claim that one runtime is universally better. OpenVINO CPU is relevant when Intel hardware,
CPU-only capacity, portability, or GPU isolation matters. TensorRT remains the NVIDIA GPU path.

OpenVINO raw alignment versus ONNX Runtime: max={ov['alignment_vs_onnxruntime']['max_abs']:.6f},
mean={ov['alignment_vs_onnxruntime']['mean_abs']:.6f},
p99={ov['alignment_vs_onnxruntime']['p99_abs']:.6f}.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
