#!/usr/bin/env python3
"""Build a YOLOv8 INT8 TensorRT engine with entropy calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from dataset_manifest import load_manifest, resolve_path


def check_cuda(result: tuple, operation: str):
    status = result[0]
    if status != cudart.cudaError_t.cudaSuccess:
        _, name = cudart.cudaGetErrorName(status)
        _, message = cudart.cudaGetErrorString(status)
        raise RuntimeError(f"{operation} failed: {name.decode()} ({message.decode()})")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def image_paths(directory: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = sorted(path for path in directory.rglob("*") if path.suffix.lower() in suffixes)
    if not paths:
        raise FileNotFoundError(f"no calibration images found under {directory}")
    return paths


def letterbox(image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    scale = min(width / image_bgr.shape[1], height / image_bgr.shape[0])
    resized_width = max(1, min(width, int(round(image_bgr.shape[1] * scale))))
    resized_height = max(1, min(height, int(round(image_bgr.shape[0] * scale))))
    resized = cv2.resize(image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    output = np.full((height, width, 3), 114, dtype=np.uint8)
    pad_left = (width - resized_width) // 2
    pad_top = (height - resized_height) // 2
    output[pad_top:pad_top + resized_height, pad_left:pad_left + resized_width] = resized
    return output


def preprocess(path: Path, input_shape: tuple[int, int, int, int]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read calibration image: {path}")
    _, channels, height, width = input_shape
    if channels != 3:
        raise ValueError(f"expected 3-channel input, got shape {input_shape}")
    letterboxed = letterbox(image, (width, height))
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor)


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self,
                 paths: list[Path],
                 input_name: str,
                 input_shape: tuple[int, int, int, int],
                 cache_path: Path,
                 cache_key: str):
        super().__init__()
        self.paths = paths
        self.input_name = input_name
        self.input_shape = input_shape
        self.cache_path = cache_path
        self.metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
        self.cache_key = cache_key
        self.index = 0
        self.host_batch = np.empty(input_shape, dtype=np.float32)
        self.device_batch = check_cuda(
            cudart.cudaMalloc(self.host_batch.nbytes), "cudaMalloc(calibration)"
        )

    def get_batch_size(self) -> int:
        return self.input_shape[0]

    def get_batch(self, names: list[str]) -> list[int] | None:
        if self.index >= len(self.paths):
            return None
        if names and names[0] != self.input_name:
            raise RuntimeError(f"calibrator expected input {self.input_name}, got {names}")
        np.copyto(self.host_batch, preprocess(self.paths[self.index], self.input_shape))
        self.index += 1
        check_cuda(
            cudart.cudaMemcpy(
                self.device_batch,
                self.host_batch.ctypes.data,
                self.host_batch.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            ),
            "cudaMemcpy(calibration)",
        )
        return [int(self.device_batch)]

    def read_calibration_cache(self) -> bytes | None:
        if self.cache_path.is_file() and self.metadata_path.is_file():
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            if metadata.get("cache_key") != self.cache_key:
                print("Calibration inputs changed; ignoring stale calibration cache.")
                return None
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)
        self.metadata_path.write_text(
            json.dumps({"schema_version": 1, "cache_key": self.cache_key}, indent=2),
            encoding="utf-8",
        )

    def close(self) -> None:
        if self.device_batch is not None:
            check_cuda(cudart.cudaFree(self.device_batch), "cudaFree(calibration)")
            self.device_batch = None


def parse_shape(text: str) -> tuple[int, int, int, int]:
    dims = tuple(int(part) for part in text.lower().split("x"))
    if len(dims) != 4 or any(dim <= 0 for dim in dims):
        raise argparse.ArgumentTypeError("shape must look like 1x3x640x640")
    return dims  # type: ignore[return-value]


def calibration_inputs(args: argparse.Namespace) -> tuple[list[Path], list[str]]:
    if args.calibration_dir is not None:
        paths = image_paths(args.calibration_dir)
        return paths, [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    document = load_manifest(args.manifest)
    records = [record for record in document["records"] if record["split"] == "calibration"]
    if not records:
        raise ValueError("dataset manifest contains no calibration records")
    return (
        [resolve_path(args.manifest, record["image"]) for record in records],
        [record["image_sha256"] for record in records],
    )


def cache_key(args: argparse.Namespace, calibration_hashes: list[str]) -> str:
    payload = {
        "onnx_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
        "calibration_image_sha256": calibration_hashes,
        "input_shape": args.input_shape,
        "preprocess": "letterbox114-bgr2rgb-fp32-div255-nchw-v1",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_engine(args: argparse.Namespace) -> None:
    check_cuda(cudart.cudaSetDevice(0), "cudaSetDevice")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(args.onnx.read_bytes()):
        for index in range(parser.num_errors):
            print(parser.get_error(index))
        raise RuntimeError(f"failed to parse ONNX: {args.onnx}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.INT8)
    if args.enable_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = args.input_shape
    if any(dim < 0 for dim in tuple(input_tensor.shape)):
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, input_shape, input_shape, input_shape)
        config.add_optimization_profile(profile)

    calibration_paths, calibration_hashes = calibration_inputs(args)
    calibrator = EntropyCalibrator(
        calibration_paths, input_name, input_shape, args.cache,
        cache_key(args, calibration_hashes),
    )
    config.int8_calibrator = calibrator
    try:
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT failed to build the INT8 engine")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(bytes(serialized))
    finally:
        calibrator.close()

    print(f"Calibration images: {len(calibration_paths)}")
    print(f"Calibration cache: {args.cache}")
    print(f"INT8 engine: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx", type=Path, default=Path("../05_torch_to_onnx/outputs/yolov8n.onnx")
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/dataset_manifest.json"))
    parser.add_argument(
        "--calibration-dir", type=Path,
        help="Override the manifest for an exploratory build."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/yolov8n_static_int8.engine"))
    parser.add_argument(
        "--cache", type=Path, default=Path("outputs/yolov8n_int8_calibration.cache")
    )
    parser.add_argument("--input-shape", type=parse_shape, default=(1, 3, 640, 640))
    parser.add_argument("--workspace-mib", type=int, default=2048)
    parser.add_argument(
        "--enable-fp16", action="store_true",
        help="Allow FP16 fallback tactics while building INT8."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workspace_mib <= 0:
        raise ValueError("--workspace-mib must be positive")
    if args.input_shape[0] != 1:
        raise ValueError("this lesson calibrator currently requires batch size 1")
    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX model not found: {args.onnx}")
    if args.calibration_dir is None and not args.manifest.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {args.manifest}")
    build_engine(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
