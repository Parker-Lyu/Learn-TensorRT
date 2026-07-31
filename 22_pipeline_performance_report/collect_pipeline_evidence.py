#!/usr/bin/env python3
"""Collect schema-v3 evidence directly from the integrated Lesson 21 executable."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], timeout: float = 120.0,
        environment: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.monotonic()
    env = os.environ.copy()
    if environment:
        env.update(environment)
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                timeout=timeout, check=False, env=env)
        return {"command": command, "returncode": result.returncode,
                "elapsed_seconds": time.monotonic() - started,
                "stdout": result.stdout, "stderr": result.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as error:
        return {"command": command, "returncode": None,
                "elapsed_seconds": time.monotonic() - started,
                "stdout": error.stdout or "", "stderr": error.stderr or "",
                "timed_out": True}
    except OSError as error:
        return {"command": command, "returncode": None,
                "elapsed_seconds": time.monotonic() - started,
                "stdout": "", "stderr": str(error), "timed_out": False}


def key_values(line: str) -> dict[str, float]:
    return {key: float(value) for key, value in
            re.findall(r"([a-zA-Z0-9_]+)=([0-9.]+)", line)}


def parse_single(text: str) -> dict[str, float]:
    values = key_values(text)
    required = {"captured", "processed", "dropped", "queue_peak", "fps"}
    if not required <= values.keys():
        raise ValueError("single-stream output is missing metrics")
    return values


def read_metrics(directory: Path) -> dict[str, Any] | None:
    path = directory / "metrics.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            result.append(value)
    return result


def merge_pipeline_snapshots(samples: list[dict[str, Any]],
                             snapshots: list[dict[str, Any]]) -> None:
    if not snapshots:
        return
    for sample in samples:
        nearest = min(snapshots, key=lambda item: abs(
            float(item.get("elapsed_seconds", 0.0)) - float(sample["elapsed_seconds"])))
        for key in ("queue_depth", "queue_peak", "captured", "submitted", "completed",
                    "evicted", "available_slots", "errors"):
            sample[key] = nearest.get(key)


def integrated_run(executable: Path, engine: Path, sources: str, frames: int, batch: int,
                   slots: int, output: Path, overload: str = "block", queue: int = 4,
                   scheduling: str = "round-robin", timeout: float = 120.0,
                   environment: dict[str, str] | None = None,
                   extra: list[str] | None = None) -> dict[str, Any]:
    command = [str(executable), str(engine), sources, str(frames), str(batch), str(slots),
               str(output), overload, str(queue), "0", scheduling]
    if extra:
        command.extend(extra)
    result = run(command, timeout=timeout, environment=environment)
    result["metrics"] = read_metrics(output)
    return result


def gpu_process_table() -> dict[int, float]:
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False)
    if query.returncode != 0:
        return {}
    result = {}
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0].isdigit():
            try:
                result[int(fields[0])] = float(fields[1])
            except ValueError:
                pass
    return result


def gpu_memory_for_process(local_pid: int, baseline_pids: set[int]) -> float | None:
    processes = gpu_process_table()
    if local_pid in processes:
        return processes[local_pid]
    # NVIDIA tooling may report the host PID while /proc exposes a container PID. Attribute memory
    # only when exactly one new GPU process appeared after launch; never sum all compute apps.
    new_processes = [memory for pid, memory in processes.items() if pid not in baseline_pids]
    return new_processes[0] if len(new_processes) == 1 else None


def process_rss_mib(pid: int) -> float | None:
    try:
        text = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"VmRSS:\s+(\d+)", text)
    return int(match.group(1)) / 1024.0 if match else None


def monitored_run(command: list[str], sample_interval: float = 1.0,
                  timeout: float | None = None) -> dict[str, Any]:
    started = time.monotonic()
    baseline_gpu_pids = set(gpu_process_table())
    process = subprocess.Popen(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    samples: list[dict[str, float | None]] = []
    timed_out = False
    while process.poll() is None:
        elapsed = time.monotonic() - started
        samples.append({"elapsed_seconds": elapsed,
                        "rss_mib": process_rss_mib(process.pid),
                        "device_memory_mib": gpu_memory_for_process(
                            process.pid, baseline_gpu_pids)})
        if timeout is not None and elapsed > timeout:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            break
        time.sleep(sample_interval)
    stdout, stderr = process.communicate()
    return {"command": command, "returncode": process.returncode,
            "elapsed_seconds": time.monotonic() - started, "stdout": stdout,
            "stderr": stderr, "timed_out": timed_out, "pid": process.pid,
            "samples": samples}


def window_trend(samples: list[dict[str, Any]], field: str, warmup_seconds: float,
                 window_seconds: float, threshold_percent: float) -> dict[str, Any]:
    usable = [(float(item["elapsed_seconds"]), item.get(field)) for item in samples
              if float(item["elapsed_seconds"]) >= warmup_seconds and item.get(field) is not None]
    if len(usable) < 2:
        return {"sample_count": len(usable), "growth_percent": None,
                "threshold_percent": threshold_percent, "warmup_seconds": warmup_seconds,
                "window_seconds": window_seconds, "available": False}
    first_start = usable[0][0]
    last_end = usable[-1][0]
    first = [float(value) for timestamp, value in usable
             if timestamp <= first_start + window_seconds]
    last = [float(value) for timestamp, value in usable
            if timestamp >= max(first_start, last_end - window_seconds)]
    first_median = statistics.median(first)
    last_median = statistics.median(last)
    growth = 0.0 if first_median == 0.0 else (last_median - first_median) / first_median * 100.0
    return {"sample_count": len(usable), "growth_percent": growth,
            "threshold_percent": threshold_percent, "warmup_seconds": warmup_seconds,
            "window_seconds": window_seconds, "first_window_median_mib": first_median,
            "last_window_median_mib": last_median, "available": True}


def cleanup_fault(result: dict[str, Any]) -> dict[str, Any]:
    result["cleanup_complete"] = result.get("returncode") not in (None, 0) and not result["timed_out"]
    return result


def platform_identity(metrics: dict[str, Any] | None) -> dict[str, Any]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version",
         "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False)
    fields = [field.strip() for field in gpu.stdout.strip().split(",")]
    environment = (metrics or {}).get("environment", {})
    return {
        "development_image": "nvcr.io/nvidia/pytorch:25.11-py3",
        "gpu": fields[0] if len(fields) > 0 else environment.get("gpu"),
        "compute_capability": fields[1] if len(fields) > 1 else environment.get("compute_capability"),
        "driver": fields[2] if len(fields) > 2 else None,
        "tensorrt": environment.get("tensorrt"),
        "cuda_runtime": environment.get("cuda_runtime"),
        "cuda_driver": environment.get("cuda_driver"),
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }


def sanitizer_evidence(gpu_smoke: Path, integrated: Path, engine: Path, image: Path,
                       output_root: Path,
                       run_tools: bool) -> dict[str, Any]:
    if not run_tools:
        missing = {"returncode": None, "tool_started": False,
                   "stderr": "not requested"}
        return {"compute_memcheck_lesson21": dict(missing),
                "lesson21_cpu_tsan": dict(missing)}
    prefix = ["compute-sanitizer", "--tool", "memcheck", "--error-exitcode", "99"]
    memcheck_cases = {
        "two_slot_batch4": run(prefix + [str(gpu_smoke), str(engine), str(image),
                                          "4", "--two-slots"], timeout=600),
        "overload": run(prefix + [str(integrated), str(engine), str(image), "64", "1", "1",
                                  str(output_root / "sanitizer-overload"), "drop-oldest", "1",
                                  "0", "round-robin"], timeout=600),
        "abort_cleanup": run(prefix + [str(integrated), str(engine), str(image), "32", "4", "2",
                                       str(output_root / "sanitizer-abort")], timeout=600,
                             environment={"LESSON21_ABORT_AFTER_SUBMISSIONS": "1"}),
    }
    expected_codes = {"two_slot_batch4": 0, "overload": 0, "abort_cleanup": 1}
    def clean(case: dict[str, Any], expected: int) -> bool:
        diagnostics = str(case.get("stdout", "")) + str(case.get("stderr", ""))
        return case.get("returncode") == expected and "ERROR SUMMARY: 0 errors" in diagnostics
    memcheck = {"cases": memcheck_cases,
                "tool_started": all(case.get("returncode") is not None and
                                    not case.get("timed_out", False)
                                    for case in memcheck_cases.values())}
    memcheck["returncode"] = 0 if all(
        clean(case, expected_codes[name]) for name, case in memcheck_cases.items()) else 1
    saved_tsan = ROOT / "22_pipeline_performance_report/outputs/tsan.json"
    if saved_tsan.is_file():
        try:
            tsan = json.loads(saved_tsan.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            tsan = {"returncode": None, "stderr": "malformed saved TSAN evidence"}
    else:
        tsan_binary = ROOT / "21_integrated_tensorrt_video_pipeline/build-tsan/integrated_frame_scheduler_tests"
        tsan = run([str(tsan_binary)]) if tsan_binary.is_file() else {
            "returncode": None, "stderr": "Lesson 21 TSAN build not found", "timed_out": False}
    diagnostics = str(tsan.get("stderr", ""))
    tsan["tool_started"] = tsan.get("returncode") is not None and \
        "unexpected memory mapping" not in diagnostics and not tsan.get("timed_out", False)
    return {"compute_memcheck_lesson21": memcheck, "lesson21_cpu_tsan": tsan}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soak-minutes", type=float, default=0.05)
    parser.add_argument("--restart-cycles", type=int, default=3)
    parser.add_argument("--run-sanitizers", action="store_true")
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "22_pipeline_performance_report/outputs/evidence.json")
    args = parser.parse_args()
    if args.soak_minutes <= 0 or args.restart_cycles <= 0 or args.sample_interval_seconds <= 0:
        parser.error("soak, restart cycles, and sample interval must be positive")

    lesson = ROOT / "21_integrated_tensorrt_video_pipeline"
    executable = lesson / "build/integrated_tensorrt_video_pipeline_gpu"
    gpu_smoke = lesson / "build/integrated_tensorrt_gpu_smoke"
    engine = ROOT / "17_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine"
    image = ROOT / "assets/img.jpeg"
    output_root = lesson / "output/report_run"
    for required in (executable, gpu_smoke, engine, image):
        if not required.is_file():
            raise FileNotFoundError(f"required Lesson 21 artifact is missing: {required}")

    batches = {}
    for batch in (1, 2, 4):
        batches[str(batch)] = integrated_run(
            executable, engine, str(image), batch * 2, batch, 2,
            output_root / f"batch-{batch}")

    overlap = run([str(gpu_smoke), str(engine), str(image), "4", "--two-slots"])
    overlap["overlap_observed"] = "submitted_before_collection=2" in overlap["stdout"]

    batch_reference = run([sys.executable,
                           str(lesson / "tests/test_batch_consistency.py"),
                           str(executable), str(engine), str(image)], timeout=180)
    batch_reference["within_tolerance"] = batch_reference["returncode"] == 0
    cuda_reference_binary = ROOT / "20_cuda_preprocess_npp/build/cuda_preprocess_tests"
    cuda_reference = run([str(cuda_reference_binary)]) if cuda_reference_binary.is_file() else {
        "command": [str(cuda_reference_binary)], "returncode": None,
        "stderr": "Lesson 20 preprocessing test is not built", "stdout": "",
        "timed_out": False}
    cuda_reference["within_tolerance"] = cuda_reference.get("returncode") == 0

    multi = integrated_run(executable, engine, f"{image},{image}", 32, 4, 2,
                           output_root / "multi")
    multi["producer"] = "lesson21"

    policy_arguments = {
        "block": ("block", 4, "round-robin"),
        "drop_oldest": ("drop-oldest", 1, "round-robin"),
        "latest_first": ("drop-oldest", 2, "latest-first"),
    }
    policies = {}
    for name, (overload, capacity, scheduling) in policy_arguments.items():
        item = integrated_run(executable, engine, f"{image},{image}", 200, 1, 1,
                              output_root / f"policy-{name}", overload, capacity, scheduling)
        metrics = item.get("metrics") or {}
        item["bounded"] = metrics.get("queue_peak", capacity + 1) <= capacity and \
            metrics.get("slots", 2) <= 1
        policies[name] = item

    fault_specs = {
        "source_read": ({"LESSON21_FAIL_SOURCE_FRAME": "2"}, 8, 4),
        "insufficient_capacity": ({"LESSON21_FAIL_INSUFFICIENT_CAPACITY": "1"}, 8, 4),
        "tensor_address": ({"LESSON21_FAIL_TENSOR_ADDRESS": "1"}, 8, 4),
        "enqueue": ({"LESSON21_FAIL_ENQUEUE": "1"}, 8, 4),
        "postprocess": ({"LESSON21_FAIL_POSTPROCESS_BATCH": "0"}, 8, 4),
        "abort_pending": ({"LESSON21_ABORT_AFTER_SUBMISSIONS": "1"}, 32, 4),
    }
    faults = {}
    for name, (environment, frames, batch) in fault_specs.items():
        faults[name] = cleanup_fault(integrated_run(
            executable, engine, str(image), frames, batch, 2, output_root / f"fault-{name}",
            timeout=30, environment=environment))
    faults["invalid_shape"] = cleanup_fault(
        run([str(gpu_smoke), str(engine), str(image), "5"], timeout=30))

    restart_started = time.monotonic()
    restart_failures = 0
    for cycle in range(args.restart_cycles):
        result = integrated_run(executable, engine, str(image), 8, 2, 2,
                                output_root / f"restart-{cycle}", timeout=30)
        restart_failures += int(result["returncode"] != 0)
    restarts = {"requested": args.restart_cycles, "failures": restart_failures,
                "elapsed_seconds": time.monotonic() - restart_started}

    duration_seconds = max(1, int(math.ceil(args.soak_minutes * 60.0)))
    soak_output = output_root / "long-lived-soak"
    soak_command = [str(executable), str(engine), str(image), "16", "4", "2",
                    str(soak_output), "block", "4", "0", "round-robin",
                    "--duration-seconds", str(duration_seconds), "--repeat-source",
                    "--metrics-interval-seconds", "1"]
    soak = monitored_run(soak_command, args.sample_interval_seconds,
                         timeout=duration_seconds + 60)
    soak["metrics"] = read_metrics(soak_output)
    merge_pipeline_snapshots(
        soak["samples"], read_json_lines(soak_output / "metrics_snapshots.jsonl"))
    soak["single_process"] = True
    soak["actual_seconds"] = soak["elapsed_seconds"]
    soak["failures"] = int(soak["returncode"] != 0)
    soak["formal_requested"] = args.soak_minutes >= 30.0

    warmup = 60.0 if soak["formal_requested"] else max(0.0, duration_seconds * 0.1)
    window = 120.0 if soak["formal_requested"] else max(1.0, duration_seconds * 0.4)
    memory = {
        "formal_requested": soak["formal_requested"],
        "host": window_trend(soak["samples"], "rss_mib", warmup, window, 5.0),
        "device": window_trend(soak["samples"], "device_memory_mib", warmup, window, 5.0),
    }

    evidence = {
        "schema_version": 3,
        "platform": platform_identity(batches["1"].get("metrics")),
        "load_matrix": {"batches": batches, "two_slot_overlap": overlap},
        "reference_checks": {"batch_vs_single": batch_reference,
                             "cpu_vs_cuda_preprocess": cuda_reference},
        "multi_stream": multi,
        "policies": policies,
        "faults": faults,
        "restarts": restarts,
        "long_lived_soak": soak,
        "memory_trend": memory,
        "sanitizers": sanitizer_evidence(
            gpu_smoke, executable, engine, image, output_root, args.run_sanitizers),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
