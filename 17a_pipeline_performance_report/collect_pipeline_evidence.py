#!/usr/bin/env python3
"""Collect bounded-pipeline, lifecycle, fault, memory, and CUDA evidence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], timeout: float = 120.0) -> dict:
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                            timeout=timeout, check=False)
    return {"command": command, "returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "stdout": result.stdout, "stderr": result.stderr}


def key_values(line: str) -> dict[str, float]:
    values = {}
    for key, value in re.findall(r"([a-zA-Z0-9_]+)=([0-9.]+)", line):
        values[key] = float(value)
    return values


def parse_single(text: str) -> dict:
    values = key_values(text)
    required = {"captured", "processed", "dropped", "queue_peak", "fps", "p50_ms", "p90_ms", "p99_ms"}
    if not required <= values.keys():
        raise ValueError("single-stream output is missing metrics")
    return values


def parse_multi(text: str) -> dict:
    lines = [line for line in text.splitlines() if line.strip()]
    total = key_values(lines[0])
    streams = [key_values(line) for line in lines[1:] if line.startswith("stream=")]
    if "total_fps" not in total or len(streams) < 2:
        raise ValueError("multi-stream output is missing metrics")
    return {"total_fps": total["total_fps"], "streams": streams}


def platform_identity() -> dict:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False)
    tensorrt = subprocess.run(
        [sys.executable, "-c", "import tensorrt; print(tensorrt.__version__)"],
        text=True, capture_output=True, check=False)
    return {
        "development_image": "nvcr.io/nvidia/pytorch:25.11-py3",
        "gpu_query": gpu.stdout.strip(),
        "gpu_query_returncode": gpu.returncode,
        "tensorrt_version": tensorrt.stdout.strip(),
        "tensorrt_query_returncode": tensorrt.returncode,
    }


def gpu_memory_mib() -> float:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=False)
    values = [float(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
    return sum(values)


def monitored_run(command: list[str]) -> dict:
    gpu_start = gpu_memory_mib()
    process = subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    rss_values = []
    gpu_values = [gpu_start]
    while process.poll() is None:
        status = Path(f"/proc/{process.pid}/status")
        if status.exists():
            match = re.search(r"VmRSS:\s+(\d+)", status.read_text(encoding="utf-8"))
            if match:
                rss_values.append(int(match.group(1)) / 1024.0)
        gpu_values.append(gpu_memory_mib())
        time.sleep(0.02)
    stdout, stderr = process.communicate()
    gpu_values.append(gpu_memory_mib())
    return {
        "command": command, "returncode": process.returncode, "stdout": stdout, "stderr": stderr,
        "host_rss_mib": {"start": rss_values[0] if rss_values else 0.0,
                         "peak": max(rss_values, default=0.0),
                         "end": rss_values[-1] if rss_values else 0.0},
        "device_memory_mib": {"start": gpu_values[0], "peak": max(gpu_values), "end": gpu_values[-1]},
    }


def require_success(result: dict, label: str) -> None:
    if result["returncode"]:
        raise RuntimeError(f"{label} failed: {result['stderr'] or result['stdout']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soak-minutes", type=float, default=0.02)
    parser.add_argument("--restart-cycles", type=int, default=100)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "17a_pipeline_performance_report/outputs/evidence.json")
    args = parser.parse_args()
    if args.soak_minutes <= 0 or args.restart_cycles < 100:
        parser.error("soak-minutes must be positive and restart-cycles must be >=100")

    single_exe = str(ROOT / "15_async_video_pipeline/build/async_video_pipeline")
    multi_exe = str(ROOT / "16_multistream_video_pipeline/build/multistream_video_pipeline")
    cuda_exe = str(ROOT / "17_cuda_preprocess_npp/build/cuda_preprocess_npp")
    cuda_test = str(ROOT / "17_cuda_preprocess_npp/build/cuda_preprocess_tests")
    for executable in (single_exe, multi_exe, cuda_exe, cuda_test):
        if not Path(executable).is_file():
            raise FileNotFoundError(f"build the lesson first: {executable}")

    single = monitored_run([single_exe, "--synthetic-frames", "500", "--queue-capacity", "4",
                            "--max-batch", "4", "--inference-ms", "8"])
    require_success(single, "single-stream run")
    single["metrics"] = parse_single(single["stdout"])
    multi = run([multi_exe, "--scheduler", "round-robin", "--queue-capacity", "4", "--max-batch", "4"])
    require_success(multi, "multi-stream run")
    multi["metrics"] = parse_multi(multi["stdout"])
    cuda = run([cuda_exe, "--iterations", "20"])
    require_success(cuda, "CUDA preprocessing benchmark")

    restart_started = time.monotonic()
    restart_failures = 0
    for _ in range(args.restart_cycles):
        result = run([single_exe, "--synthetic-frames", "8", "--queue-capacity", "2",
                      "--max-batch", "2", "--inference-ms", "0"], timeout=10)
        restart_failures += int(result["returncode"] != 0)
    restarts = {"requested": args.restart_cycles, "failures": restart_failures,
                "elapsed_seconds": time.monotonic() - restart_started}

    soak_deadline = time.monotonic() + args.soak_minutes * 60.0
    soak_cycles = soak_failures = 0
    while time.monotonic() < soak_deadline:
        result = run([single_exe, "--synthetic-frames", "1000", "--queue-capacity", "4",
                      "--max-batch", "4", "--inference-ms", "2"])
        soak_cycles += 1
        soak_failures += int(result["returncode"] != 0)
    soak = {"requested_minutes": args.soak_minutes, "cycles": soak_cycles,
            "failures": soak_failures}

    faults = {
        "invalid_input": run([single_exe, "--input", "/definitely/missing/video.mp4"]),
        "capture_failure": run([single_exe, "--synthetic-frames", "20", "--fail-capture-at", "3"]),
        "worker_failure": run([single_exe, "--synthetic-frames", "20", "--fail-worker-at", "3"]),
        "multistream_inference_failure": run([multi_exe, "--fail-inference-batch", "1"]),
    }
    for fault in faults.values():
        fault["expected_nonzero"] = fault["returncode"] != 0

    sanitizer = {
        "compute_memcheck": run(["compute-sanitizer", "--tool", "memcheck",
                                  "--error-exitcode", "99", cuda_test]),
    }
    tsan_path = ROOT / "13_cpp_producer_consumer/build-tsan/producer_consumer_tests"
    sanitizer["thread_sanitizer"] = (
        run([str(tsan_path)]) if tsan_path.is_file()
        else {"command": [str(tsan_path)], "returncode": None,
              "stdout": "", "stderr": "TSAN build not found"}
    )

    evidence = {"schema_version": 2, "platform": platform_identity(),
                "single_stream": single, "multi_stream": multi,
                "restarts": restarts, "soak": soak, "faults": faults, "sanitizers": sanitizer,
                "cuda_benchmark_csv": "17_cuda_preprocess_npp/outputs/preprocess_benchmark.csv"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
