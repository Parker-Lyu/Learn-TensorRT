#!/usr/bin/env python3
"""Send real YOLOv8 tensors to Triton and benchmark concurrent HTTP requests."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tritonclient.http as httpclient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import summarize  # noqa: E402


def infer_once(url: str, tensor: np.ndarray) -> tuple[float, tuple[int, ...], float]:
    client = httpclient.InferenceServerClient(url=url, verbose=False)
    input_tensor = httpclient.InferInput("images", list(tensor.shape), "FP32")
    input_tensor.set_data_from_numpy(tensor, binary_data=True)
    output = httpclient.InferRequestedOutput("output0", binary_data=True)
    started = time.perf_counter()
    response = client.infer("yolov8", inputs=[input_tensor], outputs=[output])
    latency_ms = (time.perf_counter() - started) * 1000.0
    values = response.as_numpy("output0")
    if values is None or values.shape != (tensor.shape[0], 84, 8400):
        raise RuntimeError(f"unexpected Triton output shape: {None if values is None else values.shape}")
    return latency_ms, values.shape, float(values.sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--input-npy", type=Path,
                        default=ROOT / "05_torch_to_onnx/outputs/input_nchw_float32.npy")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "24_triton_inference_server/outputs/client_benchmark.json")
    args = parser.parse_args()
    if args.concurrency <= 0 or args.requests < 100 or args.warmup < 0:
        parser.error("concurrency must be positive, requests >=100, and warmup non-negative")
    tensor = np.load(args.input_npy).astype(np.float32, copy=False)
    if tensor.shape != (1, 3, 640, 640):
        raise ValueError(f"expected [1,3,640,640], received {tensor.shape}")
    metadata_client = httpclient.InferenceServerClient(url=args.url, verbose=False)
    server_metadata = metadata_client.get_server_metadata()
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False)
    if gpu_query.returncode != 0:
        raise RuntimeError(f"failed to query GPU identity: {gpu_query.stderr.strip()}")
    for _ in range(args.warmup):
        infer_once(args.url, tensor)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(lambda _: infer_once(args.url, tensor), range(args.requests)))
    elapsed = time.perf_counter() - started
    evidence = {"url": args.url, "model": "yolov8",
                "environment": {"gpu": gpu_query.stdout.strip(),
                                "server_name": server_metadata["name"],
                                "server_version": server_metadata["version"]},
                "concurrency": args.concurrency,
                **summarize([item[0] for item in results], elapsed),
                "output_shape": results[-1][1], "last_output_checksum": results[-1][2]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
