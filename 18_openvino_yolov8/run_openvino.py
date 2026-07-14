#!/usr/bin/env python3
"""Run one YOLOv8 ONNX model with OpenVINO CPU in sync and async modes."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import openvino as ov

ROOT = Path(__file__).resolve().parents[1]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summary(latencies: list[float], elapsed_seconds: float) -> dict:
    if not latencies or elapsed_seconds <= 0:
        raise ValueError("latencies and elapsed time must be positive")
    return {
        "requests": len(latencies),
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": percentile(latencies, 0.50),
            "p90": percentile(latencies, 0.90),
            "p99": percentile(latencies, 0.99),
        },
        "throughput_requests_per_second": len(latencies) / elapsed_seconds,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path,
                        default=ROOT / "05_torch_to_onnx/outputs/yolov8n.onnx")
    parser.add_argument("--input-npy", type=Path,
                        default=ROOT / "05_torch_to_onnx/outputs/input_nchw_float32.npy")
    parser.add_argument("--reference-npy", type=Path,
                        default=ROOT / "05_torch_to_onnx/outputs/onnxruntime_raw_output.npy")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--async-jobs", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "18_openvino_yolov8/outputs/openvino_benchmark.json")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 100 or args.async_jobs <= 0:
        parser.error("warmup must be non-negative, iterations >=100, and async-jobs positive")

    input_tensor = np.load(args.input_npy).astype(np.float32, copy=False)
    reference = np.load(args.reference_npy).astype(np.float32, copy=False)
    core = ov.Core()
    model = core.read_model(args.onnx)
    if len(model.inputs) != 1 or len(model.outputs) != 1:
        raise RuntimeError("lesson 18 expects one input and one output")
    model_input = model.input(0)
    if list(model_input.shape) != list(input_tensor.shape):
        raise ValueError(f"input shape mismatch: model={model_input.shape}, data={input_tensor.shape}")

    latency_compiled = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
    request = latency_compiled.create_infer_request()
    for _ in range(args.warmup):
        request.infer({0: input_tensor})
    sync_latencies = []
    sync_started = time.perf_counter()
    output = None
    for _ in range(args.iterations):
        started = time.perf_counter()
        output = request.infer({0: input_tensor})[latency_compiled.output(0)]
        sync_latencies.append((time.perf_counter() - started) * 1000.0)
    sync_elapsed = time.perf_counter() - sync_started
    assert output is not None

    async_latencies: list[float] = []
    async_checksums: list[float] = []
    throughput_compiled = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
    queue = ov.AsyncInferQueue(throughput_compiled, args.async_jobs)

    def complete(infer_request, userdata):
        async_latencies.append((time.perf_counter() - userdata) * 1000.0)
        async_checksums.append(float(np.asarray(infer_request.get_output_tensor().data).sum()))

    queue.set_callback(complete)
    async_started = time.perf_counter()
    for _ in range(args.iterations):
        queue.start_async({0: input_tensor}, userdata=time.perf_counter())
    queue.wait_all()
    async_elapsed = time.perf_counter() - async_started

    error = np.abs(np.asarray(output, dtype=np.float32) - reference)
    evidence = {
        "model": str(args.onnx.relative_to(ROOT)),
        "input": str(args.input_npy.relative_to(ROOT)),
        "device": "CPU",
        "software": {"openvino": ov.__version__, "python": platform.python_version(),
                     "numpy": np.__version__},
        "compiled_properties": {
            "latency": {
                "execution_devices": list(latency_compiled.get_property("EXECUTION_DEVICES")),
                "num_streams": str(latency_compiled.get_property("NUM_STREAMS")),
            },
            "throughput": {
                "execution_devices": list(throughput_compiled.get_property("EXECUTION_DEVICES")),
                "num_streams": str(throughput_compiled.get_property("NUM_STREAMS")),
            },
        },
        "sync": summary(sync_latencies, sync_elapsed),
        "async": {**summary(async_latencies, async_elapsed), "jobs": args.async_jobs,
                  "last_checksum": async_checksums[-1]},
        "alignment_vs_onnxruntime": {
            "max_abs": float(error.max()), "mean_abs": float(error.mean()),
            "p99_abs": float(np.percentile(error, 99)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
