#!/usr/bin/env python3
"""Collect matched TensorRT 10 FP32, FP16, and INT8 trtexec evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


LESSON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = LESSON_DIR.parent
STEP_OUTPUT = LESSON_DIR / "outputs/tensorrt10"
DEFAULT_ENGINES = {
    "fp32": STEP_OUTPUT / "references/yolov8n_trt10_fp32.engine",
    "fp16": STEP_OUTPUT / "references/yolov8n_trt10_fp16.engine",
    "int8": STEP_OUTPUT / "candidate/yolov8n_qdq_int8.engine",
}
DEFAULT_EVALUATION = LESSON_DIR / "outputs/evaluation/precision_evaluation.json"
THROUGHPUT_PATTERN = re.compile(r"Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*qps")
EXPECTED_TRT_SERIES = "10.14"


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
    throughput = float(matches[-1])
    if not math.isfinite(throughput) or throughput <= 0.0:
        raise ValueError(f"trtexec reported invalid throughput: {throughput}")
    return throughput


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return (result.stdout + result.stderr).strip()


def trtexec_version(executable: str) -> str:
    output = command_output([executable, "--version"])
    match = re.search(r"TensorRT v([0-9]+)", output)
    if not match:
        raise RuntimeError("could not parse TensorRT version from trtexec output")
    digits = match.group(1)
    if len(digits) == 4:
        major, minor, patch = digits[0], digits[1], digits[2:]
    elif len(digits) >= 6:
        major, minor, patch = digits[:-4], digits[-4:-2], digits[-2:]
    else:
        raise RuntimeError("unexpected compact TensorRT version")
    return f"{int(major)}.{int(minor)}.{int(patch)}"


def require_course_tensorrt(version: str) -> None:
    if not version.startswith(EXPECTED_TRT_SERIES + "."):
        raise RuntimeError(
            f"Lesson 14 performance evidence requires TensorRT {EXPECTED_TRT_SERIES}.x, "
            f"found {version}"
        )


def load_evaluation(path: Path, engines: dict[str, Path]) -> tuple[dict, bool]:
    if not path.is_file():
        raise FileNotFoundError(f"missing precision evaluation: {path}")
    evaluation = json.loads(path.read_text(encoding="utf-8"))
    if evaluation.get("schema_version") != 1:
        raise ValueError("precision evaluation must use schema version 1")
    mapping = {"fp32": "tensorrt_fp32", "fp16": "tensorrt_fp16", "int8": "tensorrt_int8"}
    for name, evaluation_name in mapping.items():
        recorded = evaluation.get("artifacts", {}).get(evaluation_name, {}).get("sha256")
        actual = sha256(engines[name])
        if not recorded or recorded != actual:
            raise ValueError(f"{name} engine differs from the precision-evaluation artifact")
    candidate = evaluation.get("backends", {}).get("tensorrt_int8", {})
    if not isinstance(candidate.get("passed"), bool):
        raise ValueError("precision evaluation has no INT8 quality-gate result")
    return evaluation, bool(candidate["passed"])


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
    parser.add_argument("--trtexec", default=shutil.which("trtexec") or "/opt/tensorrt/bin/trtexec")
    parser.add_argument("--output-dir", type=Path, default=STEP_OUTPUT / "performance")
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
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
    _, int8_eligible = load_evaluation(args.evaluation, engines)
    version = trtexec_version(args.trtexec)
    require_course_tensorrt(version)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 3,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "tool": args.trtexec,
            "warmup_ms": args.warmup_ms,
            "iterations": args.iterations,
            "duration_sec": 0,
            "data_transfers": True,
            "inference_streams": 1,
            "throughput_source": "trtexec wall-time qps",
            "synchronization": "trtexec per-inference latency with H2D, compute, and D2H complete",
        },
        "environment": {
            "gpu": gpu_state()["output"],
            "trtexec": version,
        },
        "quality_gate": {
            "evaluation": str(args.evaluation.resolve()),
            "evaluation_sha256": sha256(args.evaluation),
            "int8_eligible_for_performance": int8_eligible,
        },
        "backends": {},
    }
    selected = {"fp32": engines["fp32"], "fp16": engines["fp16"]}
    if int8_eligible:
        selected["int8"] = engines["int8"]
    else:
        print("INT8 failed the quality gate; collecting FP32/FP16 evidence only.")
    for name, engine in selected.items():
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
        report["backends"][name] = {
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
        print(f"Benchmarked {name}: {report['backends'][name]['throughput_qps']:.3f} qps")
    report["environment"]["gpu_after"] = gpu_state()["output"]
    report_path = output_dir / "performance.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Performance report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
