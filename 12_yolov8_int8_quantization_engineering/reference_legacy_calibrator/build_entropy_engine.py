#!/usr/bin/env python3
"""Reference implementation of TensorRT's legacy entropy-calibrator API."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

LESSON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LESSON_DIR))
from calibration_preprocessing import preprocess  # noqa: E402
from dataset_manifest import DEFAULT_COCO_MANIFEST, load_manifest, resolve_path  # noqa: E402

EXPECTED_TRT_VERSION = "10.14.1.48"
INPUT_SHAPE = (1, 3, 640, 640)


def check_cuda(result: tuple, operation: str):
    if result[0] != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{operation} failed: {result[0]}")
    return result[1] if len(result) > 1 else None


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """Single-image calibrator with an identity-bound cache."""

    def __init__(self, images: list[Path], cache: Path, cache_key: str) -> None:
        super().__init__()
        if not images:
            raise ValueError("calibration image list is empty")
        self.images = images
        self.cache = cache
        self.cache_metadata = cache.with_suffix(cache.suffix + ".json")
        self.cache_key = cache_key
        self.index = 0
        self.host = np.empty(INPUT_SHAPE, dtype=np.float32)
        self.device = check_cuda(cudart.cudaMalloc(self.host.nbytes), "cudaMalloc")

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names: list[str]) -> list[int] | None:
        if self.index == len(self.images):
            return None
        np.copyto(self.host, preprocess(self.images[self.index], INPUT_SHAPE))
        self.index += 1
        check_cuda(cudart.cudaMemcpy(self.device, self.host.ctypes.data, self.host.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice), "cudaMemcpy")
        return [int(self.device)]

    def read_calibration_cache(self) -> bytes | None:
        if not self.cache.is_file() or not self.cache_metadata.is_file():
            return None
        metadata = json.loads(self.cache_metadata.read_text(encoding="utf-8"))
        return self.cache.read_bytes() if metadata.get("cache_key") == self.cache_key else None

    def write_calibration_cache(self, data: bytes) -> None:
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache.write_bytes(data)
        self.cache_metadata.write_text(json.dumps({"schema_version": 1,
            "cache_key": self.cache_key}, indent=2) + "\n", encoding="utf-8")

    def close(self) -> None:
        if self.device is not None:
            check_cuda(cudart.cudaFree(self.device), "cudaFree")
            self.device = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_COCO_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--workspace-mib", type=int, default=2048)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if trt.__version__ != EXPECTED_TRT_VERSION:
        raise RuntimeError(f"requires TensorRT {EXPECTED_TRT_VERSION}, found {trt.__version__}")
    manifest = load_manifest(args.manifest)
    records = [r for r in manifest["records"] if r["split"] == "calibration"]
    images = [resolve_path(args.manifest, r["image"]) for r in records]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing calibration images; first missing path: {missing[0]}")
    cache = args.cache or args.output.with_suffix(".cache")
    key = hashlib.sha256((sha256(args.onnx) + sha256(args.manifest)).encode()).hexdigest()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        raise RuntimeError("failed to parse ONNX: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.INT8)
    calibrator = EntropyCalibrator(images, cache, key)
    config.int8_calibrator = calibrator
    try:
        serialized = builder.build_serialized_network(network, config)
    finally:
        calibrator.close()
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the entropy-calibrated engine")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(bytes(serialized))
    metadata = {"schema_version": 1, "name": "tensorrt_int8",
        "build_mode": "implicit-int8-entropy-calibrator-reference",
        "calibration_algorithm": "entropy", "tensorrt_version": trt.__version__,
        "manifest_sha256": sha256(args.manifest), "engine_sha256": sha256(args.output)}
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
