#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import time
from pathlib import Path

import numpy as np

from tiny_llm import ModelConfig, TinyTransformer, tokenize

ROOT = Path(__file__).resolve().parent


def run_once(model, input_ids, output_length):
    batch, input_length = input_ids.shape
    cache = model.new_cache(batch)
    started = time.perf_counter()
    logits = None
    for position in range(input_length):
        logits = model.step(input_ids[:, position], cache)
    token = np.argmax(logits, axis=-1)
    first_token_at = time.perf_counter()
    decode_times = []
    for _ in range(output_length - 1):
        step_started = time.perf_counter()
        logits = model.step(token, cache)
        token = np.argmax(logits, axis=-1)
        decode_times.append(time.perf_counter() - step_started)
    finished = time.perf_counter()
    ttft = first_token_at - started
    decode_total = sum(decode_times)
    return {"ttft_ms": ttft * 1000.0,
            "time_per_output_token_ms": statistics.fmean(decode_times) * 1000.0,
            "prefill_tokens_per_second": batch * input_length / ttft,
            "decode_tokens_per_second": batch * (output_length - 1) / decode_total,
            "total_tokens_per_second": batch * (input_length + output_length) / (finished - started)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-lengths", default="16,64")
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--output-length", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    lengths = [int(value) for value in args.input_lengths.split(",")]
    batches = [int(value) for value in args.batch_sizes.split(",")]
    if min(lengths + batches + [args.output_length, args.repetitions]) <= 0:
        parser.error("all sizes and repetitions must be positive")
    model = TinyTransformer()
    rows = []
    for input_length in lengths:
        for batch in batches:
            inputs = tokenize("TensorRT deployment interview", input_length, batch)
            for _ in range(args.warmup): run_once(model, inputs, args.output_length)
            samples = [run_once(model, inputs, args.output_length) for _ in range(args.repetitions)]
            row = {"input_length": input_length, "batch": batch,
                   "output_length": args.output_length}
            for key in samples[0]: row[key] = statistics.fmean(sample[key] for sample in samples)
            row["estimated_kv_cache_mib"] = model.kv_cache_bytes(
                batch, input_length + args.output_length) / 2**20
            rows.append(row)
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    evidence = {
        "model": {"name": "educational-tiny-transformer", "revision": model.config.revision,
                  "config": vars(model.config), "weight_format": "FP32 deterministic NumPy",
                  "weight_memory_mib": model.weight_bytes() / 2**20},
        "tokenizer": "UTF-8 byte tokenizer v1", "backend": "NumPy CPU autoregressive",
        "hardware": {"cpu_model": cpu_model, "logical_cpu_count": os.cpu_count(),
                     "peak_host_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                     "peak_gpu_memory_mib": 0.0},
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "methodology": {"warmup": args.warmup, "repetitions": args.repetitions,
                        "fixed_output_length": args.output_length}, "results": rows}
    output = ROOT / "outputs/llm_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
