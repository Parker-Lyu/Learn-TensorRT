#!/usr/bin/env python3
"""Build and identity-record the matched TensorRT 10 Step 06 engines."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import tensorrt as trt


LESSON_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = LESSON_DIR.parent
OUTPUT_DIR = LESSON_DIR / "outputs/04_modelopt_qdq/trt10"
FP32_ONNX = REPO_ROOT / "05_torch_to_onnx/outputs/yolov8n.onnx"
QDQ_ONNX = (
    LESSON_DIR
    / "outputs/04_modelopt_qdq/export/"
    "yolov8n_modelopt_qdq_native_fp16_calibration_v3.onnx"
)
EXPECTED_TRT_VERSION = "10.14.1.48"
EXPECTED_IO = [
    {"name": "images", "mode": "INPUT", "dtype": "FLOAT", "shape": [1, 3, 640, 640]},
    {"name": "output0", "mode": "OUTPUT", "dtype": "FLOAT", "shape": [1, 84, 8400]},
]
BUILD_TIME_PATTERN = re.compile(r"Engine built in\s+([0-9]+(?:\.[0-9]+)?)\s+sec")


@dataclass(frozen=True)
class BuildSpec:
    name: str
    mode: str
    onnx_path: Path
    engine_path: Path
    layer_info_path: Path
    log_path: Path
    timing_cache_path: Path
    metadata_path: Path
    flags: tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_identity(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"required source artifact not found: {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(
            f"source artifact identity changed for {path}: expected {expected_sha256}, found {actual}"
        )


def build_command(executable: str, spec: BuildSpec) -> list[str]:
    return [
        executable,
        f"--onnx={spec.onnx_path}",
        f"--saveEngine={spec.engine_path}",
        *spec.flags,
        "--builderOptimizationLevel=3",
        "--skipInference",
        "--profilingVerbosity=detailed",
        "--dumpLayerInfo",
        f"--exportLayerInfo={spec.layer_info_path}",
        f"--timingCacheFile={spec.timing_cache_path}",
    ]


def build_specs(output_dir: Path) -> list[BuildSpec]:
    references = output_dir / "references"
    candidate = output_dir / "candidate"
    reference_cache = references / "trt10_reference.timing.cache"
    return [
        BuildSpec(
            "tensorrt_fp32",
            "strongly-typed-fp32-reference",
            FP32_ONNX,
            references / "yolov8n_trt10_fp32.engine",
            references / "yolov8n_trt10_fp32.layers.json",
            references / "yolov8n_trt10_fp32.build.log",
            reference_cache,
            references / "yolov8n_trt10_fp32.engine.json",
            ("--stronglyTyped",),
        ),
        BuildSpec(
            "tensorrt_fp16",
            "fp16-reference",
            FP32_ONNX,
            references / "yolov8n_trt10_fp16.engine",
            references / "yolov8n_trt10_fp16.layers.json",
            references / "yolov8n_trt10_fp16.build.log",
            reference_cache,
            references / "yolov8n_trt10_fp16.engine.json",
            ("--fp16",),
        ),
        BuildSpec(
            "tensorrt_int8",
            "strongly-typed-native-fp16-qdq-int8",
            QDQ_ONNX,
            candidate / "yolov8n_modelopt_hp_fp16_trt10.engine",
            candidate / "yolov8n_modelopt_hp_fp16_trt10.layers.json",
            candidate / "yolov8n_modelopt_hp_fp16_trt10.build.log",
            candidate / "trt10_qdq.timing.cache",
            candidate / "yolov8n_modelopt_hp_fp16_trt10.engine.json",
            ("--stronglyTyped",),
        ),
    ]


def run_build(command: list[str], log_path: Path) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"trtexec failed with exit code {result.returncode}; see {log_path}")
    return elapsed


def inspect_engine(engine_path: Path) -> list[dict[str, object]]:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
    bindings = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        bindings.append(
            {
                "name": name,
                "mode": str(engine.get_tensor_mode(name)).split(".")[-1],
                "dtype": str(engine.get_tensor_dtype(name)).split(".")[-1],
                "shape": list(engine.get_tensor_shape(name)),
            }
        )
    if bindings != EXPECTED_IO:
        raise ValueError(f"unexpected TensorRT engine I/O contract: {bindings}")
    return bindings


def require_artifacts(spec: BuildSpec) -> None:
    for path in (spec.engine_path, spec.layer_info_path, spec.timing_cache_path, spec.log_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"trtexec did not create a non-empty artifact: {path}")


def log_build_duration(log_path: Path) -> float | None:
    matches = BUILD_TIME_PATTERN.findall(log_path.read_text(encoding="utf-8", errors="replace"))
    return float(matches[-1]) if matches else None


def artifact(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def write_metadata(spec: BuildSpec, command: list[str], wall_duration_sec: float) -> dict:
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "name": spec.name,
        "build_mode": spec.mode,
        "tensorrt_version": trt.__version__,
        "command": command,
        "source_onnx": artifact(spec.onnx_path),
        "engine": artifact(spec.engine_path),
        "build_log": artifact(spec.log_path),
        "layer_info": artifact(spec.layer_info_path),
        "timing_cache": artifact(spec.timing_cache_path),
        "io_bindings": inspect_engine(spec.engine_path),
        "build_duration_sec": {
            "wrapper_wall_time": wall_duration_sec,
            "trtexec_engine_build": log_build_duration(spec.log_path),
        },
    }
    spec.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument(
        "--only", choices=("all", "fp32", "fp16", "int8"), default="all",
        help="Build all engines or one named engine while preserving the fixed commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if trt.__version__ != EXPECTED_TRT_VERSION:
        raise RuntimeError(
            f"Step 06 requires TensorRT {EXPECTED_TRT_VERSION}, found {trt.__version__}"
        )
    for path in (FP32_ONNX, QDQ_ONNX):
        if not path.is_file():
            raise FileNotFoundError(f"required source artifact not found: {path}")

    selected = {
        "all": {"tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8"},
        "fp32": {"tensorrt_fp32"},
        "fp16": {"tensorrt_fp16"},
        "int8": {"tensorrt_int8"},
    }[args.only]
    completed = {}
    for spec in build_specs(args.output_dir.resolve()):
        if spec.name not in selected:
            continue
        spec.engine_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_command(args.trtexec, spec)
        print(f"Building {spec.name}: {' '.join(command)}", flush=True)
        wall_duration = run_build(command, spec.log_path)
        require_artifacts(spec)
        completed[spec.name] = write_metadata(spec, command, wall_duration)
        print(f"Recorded {spec.name}: {spec.metadata_path}", flush=True)

    if args.only == "all":
        summary_path = args.output_dir.resolve() / "references/reference_builds.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "tensorrt_version": trt.__version__,
                    "builds": completed,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Build summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
