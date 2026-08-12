#!/usr/bin/env python3
"""Compare OpenVINO CPU evidence with TensorRT GPU evidence validated by lesson 15."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TENSORRT_PRECISIONS = ("fp32", "fp16", "int8")


def render_comparison(ov: dict, trt: dict) -> str:
    if trt.get("schema_version") != 3:
        raise ValueError("TensorRT performance evidence must use schema version 3")
    backends = trt.get("backends", {})
    missing = {"fp32", "fp16"} - set(backends)
    if missing:
        raise ValueError(
            "TensorRT performance evidence is missing required backends: "
            + ", ".join(sorted(missing))
        )

    async_evidence = ov["async"]
    is_current_openvino_schema = ov.get("schema_version") == 2
    async_response_latency = async_evidence.get("response_latency_ms",
                                                async_evidence.get("latency_ms"))
    if async_response_latency is None:
        raise ValueError("OpenVINO async evidence is missing response latency")
    rows = [
        ("OpenVINO CPU sync", "request wall time", ov["sync"]["latency_ms"],
         ov["sync"]["throughput_requests_per_second"]),
        ("OpenVINO CPU async", "submit-to-completion response", async_response_latency,
         async_evidence["throughput_requests_per_second"]),
    ]
    if "inference_latency_ms" in async_evidence:
        rows.append(("OpenVINO CPU async", "InferRequest execution",
                     async_evidence["inference_latency_ms"],
                     async_evidence["throughput_requests_per_second"]))
    for precision in TENSORRT_PRECISIONS:
        if precision not in backends:
            continue
        backend = backends[precision]
        rows.append((f"TensorRT GPU {precision.upper()}", "synchronized request",
                     backend["latency_ms"], backend["throughput_qps"]))
    table = "\n".join(
        f"| {name} | {scope} | {latency['mean']:.3f} | {latency['p50']:.3f} | "
        f"{latency['p90']:.3f} | {latency['p99']:.3f} | {throughput:.1f} |"
        for name, scope, latency, throughput in rows)
    int8_note = (
        ""
        if "int8" in backends
        else "\n\nTensorRT INT8 is omitted because the quality-gated canonical evidence contains no "
             "INT8 performance measurement."
    )
    openvino_note = (
        f"OpenVINO selected {async_evidence['jobs_actual']} async requests "
        f"(requested={async_evidence['jobs_requested']}, "
        f"optimal={async_evidence['optimal_number_of_infer_requests']})."
        if is_current_openvino_schema
        else "Legacy OpenVINO evidence has no InferRequest execution latency or automatic request "
             "pool metadata; rerun lesson 23 for the current measurement schema."
    )
    return f"""# OpenVINO CPU vs TensorRT GPU

## Measurement Environments

- OpenVINO CPU: `{ov['hardware']['cpu_model']}`, {ov['hardware']['logical_cpu_count']} logical CPUs, OpenVINO `{ov['software']['openvino']}`
- TensorRT GPU: `{trt['environment']['gpu']}`, TensorRT `{trt['environment']['trtexec']}`

## Results

| Backend | Latency scope | Mean ms | P50 ms | P90 ms | P99 ms | Requests/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{table}{int8_note}

{openvino_note} Async response latency includes request-pool backpressure; InferRequest execution
latency excludes the time spent waiting for an available request. The repeated async throughput is
one run-level measurement, not a separate throughput measurement for each latency scope.

These are different devices and execution modes, so the table is a deployment comparison rather
than a claim that one runtime is universally better. OpenVINO CPU is relevant when Intel hardware,
CPU-only capacity, portability, or GPU isolation matters. TensorRT remains the NVIDIA GPU path.

OpenVINO raw alignment versus ONNX Runtime: max={ov['alignment_vs_onnxruntime']['max_abs']:.6f},
mean={ov['alignment_vs_onnxruntime']['mean_abs']:.6f},
p99={ov['alignment_vs_onnxruntime']['p99_abs']:.6f}.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openvino", type=Path,
                        default=ROOT / "23_openvino_yolov8/outputs/openvino_benchmark.json")
    parser.add_argument("--tensorrt", type=Path,
                        default=ROOT / "14_yolov8_int8_quantization_engineering/outputs/"
                        "tensorrt10/performance/performance.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "23_openvino_yolov8/outputs/comparison.md")
    args = parser.parse_args()
    ov = json.loads(args.openvino.read_text(encoding="utf-8"))
    trt = json.loads(args.tensorrt.read_text(encoding="utf-8"))
    text = render_comparison(ov, trt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
