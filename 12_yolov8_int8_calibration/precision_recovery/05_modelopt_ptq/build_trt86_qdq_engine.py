#!/usr/bin/env python3
"""Build an optimized TensorRT 8.6 INT8+FP16 engine from a ModelOpt Q/DQ ONNX model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import tensorrt as trt


LESSON_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = LESSON_DIR / "outputs/precision_recovery/05_modelopt_ptq"
DEFAULT_ONNX = OUTPUT_DIR / "yolov8n_modelopt_int8_max_train3000.onnx"
DEFAULT_ENGINE = OUTPUT_DIR / "yolov8n_modelopt_int8_max_train3000_trt86_int8_fp16.engine"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_command(
    executable: str,
    onnx_path: Path,
    engine_path: Path,
    layer_info_path: Path,
    timing_cache_path: Path,
) -> list[str]:
    return [
        executable,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--int8",
        "--fp16",
        "--builderOptimizationLevel=3",
        "--skipInference",
        "--profilingVerbosity=detailed",
        "--dumpLayerInfo",
        f"--exportLayerInfo={layer_info_path}",
        f"--timingCacheFile={timing_cache_path}",
    ]


def run_build(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"trtexec failed with exit code {result.returncode}; see {log_path}")


def inspect_engine(engine_path: Path) -> list[dict[str, object]]:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
    bindings = []
    if hasattr(engine, "num_io_tensors"):
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            bindings.append({
                "name": name,
                "mode": str(engine.get_tensor_mode(name)).split(".")[-1],
                "dtype": str(engine.get_tensor_dtype(name)).split(".")[-1],
                "shape": list(engine.get_tensor_shape(name)),
            })
    else:
        for index in range(engine.num_bindings):
            bindings.append({
                "name": engine.get_binding_name(index),
                "mode": "INPUT" if engine.binding_is_input(index) else "OUTPUT",
                "dtype": str(engine.get_binding_dtype(index)).split(".")[-1],
                "shape": list(engine.get_binding_shape(index)),
            })
    expected = [
        {"name": "images", "mode": "INPUT", "dtype": "FLOAT", "shape": [1, 3, 640, 640]},
        {"name": "output0", "mode": "OUTPUT", "dtype": "FLOAT", "shape": [1, 84, 8400]},
    ]
    if bindings != expected:
        raise ValueError(f"unexpected TensorRT engine I/O contract: {bindings}")
    return bindings


def precision_evidence(layer_info_path: Path) -> dict[str, int]:
    text = layer_info_path.read_text(encoding="utf-8")
    evidence = {
        "int8_mentions": text.count("Int8") + text.count("INT8"),
        "fp16_mentions": text.count("FP16") + text.count("Half"),
        "fp32_mentions": text.count("FP32") + text.count("Float"),
    }
    if evidence["int8_mentions"] == 0:
        raise ValueError("Engine Inspector contains no INT8 precision evidence")
    if evidence["fp16_mentions"] == 0:
        raise ValueError("Engine Inspector contains no FP16 high-precision evidence")
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--trtexec", default="trtexec")
    parser.add_argument("--log", type=Path)
    parser.add_argument("--layer-info", type=Path)
    parser.add_argument("--timing-cache", type=Path)
    parser.add_argument("--metadata", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    onnx_path = args.onnx.resolve()
    engine_path = args.engine.resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"Q/DQ ONNX model not found: {onnx_path}")
    if trt.__version__ != "8.6.1":
        raise RuntimeError(f"this recovery build requires TensorRT 8.6.1, found {trt.__version__}")

    log_path = (args.log or engine_path.with_suffix(".build.log")).resolve()
    layer_info_path = (args.layer_info or engine_path.with_suffix(".layers.json")).resolve()
    timing_cache_path = (
        args.timing_cache or engine_path.parent / "modelopt_qdq_trt86.timing.cache"
    ).resolve()
    metadata_path = (args.metadata or engine_path.with_suffix(".engine.json")).resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_command(
        args.trtexec, onnx_path, engine_path, layer_info_path, timing_cache_path
    )
    run_build(command, log_path)
    for artifact in (engine_path, layer_info_path, timing_cache_path):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f"trtexec did not create expected artifact: {artifact}")

    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tensorrt_version": trt.__version__,
        "build_mode": "explicit-qdq-int8-with-fp16-high-precision",
        "command": command,
        "onnx": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "engine": str(engine_path),
        "engine_sha256": sha256(engine_path),
        "io_bindings": inspect_engine(engine_path),
        "build_log": str(log_path),
        "build_log_sha256": sha256(log_path),
        "layer_info": str(layer_info_path),
        "layer_info_sha256": sha256(layer_info_path),
        "precision_evidence": precision_evidence(layer_info_path),
        "timing_cache": str(timing_cache_path),
        "timing_cache_sha256": sha256(timing_cache_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Engine: {engine_path}")
    print(f"Layer info: {layer_info_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
