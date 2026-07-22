#!/usr/bin/env python3
"""Build a YOLOv8 INT8 TensorRT engine with a selected PTQ calibrator."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cudart

from dataset_manifest import DEFAULT_COCO_MANIFEST, load_manifest, resolve_path

PRECISION_PROFILES = {
    "none": (),
    "detection_head_fp16": (),
}

DETECTION_HEAD_PREFIXES = ("/model.22/cv2.", "/model.22/cv3.")
DETECTION_HEAD_TOWER_EXPECTED_TYPES = {
    trt.LayerType.CONVOLUTION: 18,
    trt.LayerType.ACTIVATION: 12,
    trt.LayerType.ELEMENTWISE: 12,
}
DETECTION_HEAD_EXPECTED_TYPES = {
    trt.LayerType.CONVOLUTION: 19,
    trt.LayerType.ACTIVATION: 13,
    trt.LayerType.ELEMENTWISE: 18,
    trt.LayerType.SHUFFLE: 10,
    trt.LayerType.CONCATENATION: 4,
    trt.LayerType.SLICE: 2,
    trt.LayerType.SOFTMAX: 1,
}
DETECTION_HEAD_TENSOR_PREFIX = "/model.22/"


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


class CalibrationResources:
    def __init__(self,
                 paths: list[Path],
                 input_name: str,
                 input_shape: tuple[int, int, int, int],
                 cache_path: Path,
                 cache_key: str,
                 algorithm: str):
        self.paths = paths
        self.input_name = input_name
        self.input_shape = input_shape
        self.cache_path = cache_path
        self.metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
        self.cache_key = cache_key
        self.algorithm = algorithm
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
            json.dumps({
                "schema_version": 1,
                "calibration_algorithm": self.algorithm,
                "cache_key": self.cache_key,
            }, indent=2),
            encoding="utf-8",
        )

    def close(self) -> None:
        if self.device_batch is not None:
            check_cuda(cudart.cudaFree(self.device_batch), "cudaFree(calibration)")
            self.device_batch = None


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self,
                 paths: list[Path],
                 input_name: str,
                 input_shape: tuple[int, int, int, int],
                 cache_path: Path,
                 cache_key: str):
        super().__init__()
        self.resources = CalibrationResources(
            paths, input_name, input_shape, cache_path, cache_key, "entropy"
        )

    def get_batch_size(self) -> int:
        return self.resources.get_batch_size()

    def get_batch(self, names: list[str]) -> list[int] | None:
        return self.resources.get_batch(names)

    def read_calibration_cache(self) -> bytes | None:
        return self.resources.read_calibration_cache()

    def write_calibration_cache(self, cache: bytes) -> None:
        self.resources.write_calibration_cache(cache)

    def close(self) -> None:
        self.resources.close()


class MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self,
                 paths: list[Path],
                 input_name: str,
                 input_shape: tuple[int, int, int, int],
                 cache_path: Path,
                 cache_key: str):
        super().__init__()
        self.resources = CalibrationResources(
            paths, input_name, input_shape, cache_path, cache_key, "minmax"
        )

    def get_batch_size(self) -> int:
        return self.resources.get_batch_size()

    def get_batch(self, names: list[str]) -> list[int] | None:
        return self.resources.get_batch(names)

    def read_calibration_cache(self) -> bytes | None:
        return self.resources.read_calibration_cache()

    def write_calibration_cache(self, cache: bytes) -> None:
        self.resources.write_calibration_cache(cache)

    def close(self) -> None:
        self.resources.close()


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
        "calibration_algorithm": args.calibrator,
        "input_shape": args.input_shape,
        "preprocess": "letterbox114-bgr2rgb-fp32-div255-nchw-v1",
        "tensorrt_version": trt.__version__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def create_calibrator(
    algorithm: str,
    paths: list[Path],
    input_name: str,
    input_shape: tuple[int, int, int, int],
    cache_path: Path,
    identity: str,
) -> EntropyCalibrator | MinMaxCalibrator:
    calibrator_types = {
        "entropy": EntropyCalibrator,
        "minmax": MinMaxCalibrator,
    }
    try:
        calibrator_type = calibrator_types[algorithm]
    except KeyError as error:
        raise ValueError(f"unsupported calibration algorithm: {algorithm}") from error
    return calibrator_type(paths, input_name, input_shape, cache_path, identity)


def precision_profile_layers(network, profile: str) -> list:
    try:
        required_names = PRECISION_PROFILES[profile]
    except KeyError as error:
        raise ValueError(f"unsupported precision profile: {profile}") from error

    if profile == "detection_head_fp16":
        layers = [network.get_layer(index) for index in range(network.num_layers)]
        towers = [layer for layer in layers if layer.name.startswith(DETECTION_HEAD_PREFIXES)]
        tower_counts = Counter(layer.type for layer in towers)
        if tower_counts != Counter(DETECTION_HEAD_TOWER_EXPECTED_TYPES):
            readable = {str(layer_type): count for layer_type, count in tower_counts.items()}
            raise ValueError(
                "detection_head_fp16 matched unexpected prediction-tower structure: "
                f"expected {DETECTION_HEAD_TOWER_EXPECTED_TYPES}, found {readable}"
            )

        consumers = defaultdict(list)
        for layer in layers:
            for input_index in range(layer.num_inputs):
                tensor = layer.get_input(input_index)
                if tensor is not None:
                    consumers[tensor.name].append(layer)

        selected = []
        selected_names = set()
        pending = deque(towers)
        while pending:
            layer = pending.popleft()
            if layer.name in selected_names:
                continue
            selected_names.add(layer.name)
            selected.append(layer)
            for output_index in range(layer.num_outputs):
                pending.extend(consumers[layer.get_output(output_index).name])

        type_counts = Counter(layer.type for layer in selected)
        if type_counts != Counter(DETECTION_HEAD_EXPECTED_TYPES):
            readable = {str(layer_type): count for layer_type, count in type_counts.items()}
            raise ValueError(
                "detection_head_fp16 matched an unexpected complete-head structure: "
                f"expected {DETECTION_HEAD_EXPECTED_TYPES}, found {readable}"
            )
        network_outputs = {
            network.get_output(index).name for index in range(network.num_outputs)
        }
        selected_outputs = {
            layer.get_output(index).name
            for layer in selected
            for index in range(layer.num_outputs)
        }
        if not network_outputs.issubset(selected_outputs):
            raise ValueError("detection_head_fp16 does not reach every network output")
        return selected

    if not required_names:
        return []

    layers_by_name = {}
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        if layer.name in layers_by_name:
            raise ValueError(f"network contains duplicate layer name: {layer.name}")
        layers_by_name[layer.name] = layer

    missing = [name for name in required_names if name not in layers_by_name]
    if missing:
        raise ValueError(
            f"precision profile {profile} did not match required layer(s): {missing}"
        )

    selected = []
    for name in required_names:
        layer = layers_by_name[name]
        if layer.type != trt.LayerType.CONVOLUTION:
            raise ValueError(f"precision-constrained layer is not a convolution: {name}")
        selected.append(layer)
    return selected


def apply_precision_profile(network, config, profile: str) -> list[str]:
    selected = precision_profile_layers(network, profile)
    if not selected:
        return []

    constrained = []
    for layer in selected:
        layer.precision = trt.float16
        for output_index in range(layer.num_outputs):
            layer.set_output_type(output_index, trt.float16)
        constrained.append(layer.name)

    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    return constrained


def validate_detection_head_inspector(
    layer_information: str, network_output_names: set[str]
) -> dict[str, object]:
    document = json.loads(layer_information)
    layers = document.get("Layers")
    if not isinstance(layers, list):
        raise ValueError("Engine Inspector did not return a layer list")

    checked = []
    violations = []
    boundary_conversions = []
    for layer in layers:
        tensors = layer.get("Inputs", []) + layer.get("Outputs", [])
        names = [layer.get("Name", "")] + [tensor.get("Name", "") for tensor in tensors]
        if not any(DETECTION_HEAD_TENSOR_PREFIX in name for name in names):
            continue
        outputs = layer.get("Outputs", [])
        if not outputs:
            continue
        checked.append(layer.get("Name", ""))
        for output in outputs:
            name = output.get("Name", "")
            tensor_format = output.get("Format/Datatype", "")
            if name in network_output_names:
                if "FP16" not in tensor_format:
                    boundary_conversions.append({"layer": layer.get("Name", ""), "tensor": name})
                continue
            if "FP16" not in tensor_format:
                violations.append({
                    "layer": layer.get("Name", ""),
                    "tensor": name,
                    "format": tensor_format,
                })
    if not checked:
        raise ValueError("Engine Inspector contains no detection-head execution layers")
    if violations:
        preview = ", ".join(item["layer"] for item in violations[:5])
        raise ValueError(
            f"detection-head FP16 verification failed for {len(violations)} output(s): {preview}"
        )
    return {
        "status": "PASS",
        "head_execution_layers_checked": len(checked),
        "non_fp16_internal_outputs": 0,
        "external_output_conversions": boundary_conversions,
    }


def configure_timing_cache(config, path: Path | None):
    if path is None:
        return None
    serialized = path.read_bytes() if path.is_file() else b""
    timing_cache = config.create_timing_cache(serialized)
    if timing_cache is None:
        raise RuntimeError(f"failed to create TensorRT timing cache from {path}")
    if not config.set_timing_cache(timing_cache, False):
        raise RuntimeError(f"TensorRT rejected timing cache: {path}")
    return timing_cache


def save_timing_cache(config, path: Path | None) -> None:
    if path is None:
        return
    timing_cache = config.get_timing_cache()
    if timing_cache is None:
        raise RuntimeError("TensorRT builder did not expose a timing cache after the build")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(timing_cache.serialize()))


def write_engine_evidence(
    args: argparse.Namespace,
    serialized_engine: bytes,
    logger: trt.Logger,
    calibration_count: int,
    identity: str,
    constrained_layers: list[str],
    network_output_names: set[str],
) -> None:
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    if engine is None:
        raise RuntimeError("failed to deserialize the newly built engine for inspection")
    inspector = engine.create_engine_inspector()
    layer_information = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    inspector_path = args.output.with_suffix(args.output.suffix + ".layers.json")
    inspector_path.write_text(layer_information + "\n", encoding="utf-8")
    precision_verification = None
    if args.precision_profile == "detection_head_fp16":
        precision_verification = validate_detection_head_inspector(
            layer_information, network_output_names
        )

    metadata = {
        "schema_version": 1,
        "onnx": str(args.onnx.resolve()),
        "onnx_sha256": hashlib.sha256(args.onnx.read_bytes()).hexdigest(),
        "manifest": None if args.calibration_dir else str(args.manifest.resolve()),
        "manifest_sha256": (
            None if args.calibration_dir else hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        ),
        "calibration_images": calibration_count,
        "calibration_algorithm": args.calibrator,
        "calibration_cache": str(args.cache.resolve()),
        "calibration_cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
        "calibration_cache_key": identity,
        "precision_profile": args.precision_profile,
        "precision_constraint_flag": (
            "OBEY_PRECISION_CONSTRAINTS" if constrained_layers else None
        ),
        "constrained_layers": [
            {"name": name, "compute_precision": "FP16", "output_type": "FP16"}
            for name in constrained_layers
        ],
        "precision_verification": precision_verification,
        "input_shape": list(args.input_shape),
        "workspace_mib": args.workspace_mib,
        "fp16_fallback_enabled": args.enable_fp16,
        "tensorrt_version": trt.__version__,
        "timing_cache": None if args.timing_cache is None else str(args.timing_cache.resolve()),
        "timing_cache_sha256": (
            None
            if args.timing_cache is None
            else hashlib.sha256(args.timing_cache.read_bytes()).hexdigest()
        ),
        "engine": str(args.output.resolve()),
        "engine_sha256": hashlib.sha256(serialized_engine).hexdigest(),
        "engine_inspector": str(inspector_path.resolve()),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


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
    constrained_layers = apply_precision_profile(network, config, args.precision_profile)
    network_output_names = {
        network.get_output(index).name for index in range(network.num_outputs)
    }
    configure_timing_cache(config, args.timing_cache)

    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = args.input_shape
    if any(dim < 0 for dim in tuple(input_tensor.shape)):
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, input_shape, input_shape, input_shape)
        config.add_optimization_profile(profile)

    calibration_paths, calibration_hashes = calibration_inputs(args)
    identity = cache_key(args, calibration_hashes)
    calibrator = create_calibrator(
        args.calibrator,
        calibration_paths,
        input_name,
        input_shape,
        args.cache,
        identity,
    )
    config.int8_calibrator = calibrator
    try:
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT failed to build the INT8 engine")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        serialized_bytes = bytes(serialized)
        args.output.write_bytes(serialized_bytes)
        save_timing_cache(config, args.timing_cache)
        write_engine_evidence(
            args,
            serialized_bytes,
            logger,
            len(calibration_paths),
            identity,
            constrained_layers,
            network_output_names,
        )
    finally:
        calibrator.close()

    print(f"Calibration images: {len(calibration_paths)}")
    print(f"Calibration algorithm: {args.calibrator}")
    print(f"Precision profile: {args.precision_profile}")
    print(f"Constrained layers: {len(constrained_layers)}")
    if args.timing_cache is not None:
        print(f"Timing cache: {args.timing_cache}")
    print(f"Calibration cache: {args.cache}")
    print(f"INT8 engine: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx", type=Path, default=Path("../05_torch_to_onnx/outputs/yolov8n.onnx")
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_COCO_MANIFEST)
    parser.add_argument(
        "--calibration-dir", type=Path,
        help="Override the manifest for an exploratory build."
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/yolov8n_static_int8.engine"))
    parser.add_argument(
        "--cache", type=Path, default=Path("outputs/yolov8n_int8_calibration.cache")
    )
    parser.add_argument(
        "--calibrator",
        choices=("entropy", "minmax"),
        default="entropy",
        help="PTQ calibration algorithm; use separate cache files for each algorithm.",
    )
    parser.add_argument(
        "--precision-profile",
        choices=tuple(PRECISION_PROFILES),
        default="none",
        help="Named layer-precision constraints applied after parsing the ONNX network.",
    )
    parser.add_argument(
        "--timing-cache",
        type=Path,
        help="Persistent TensorRT tactic timing cache reused across compatible engine builds.",
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
        raise FileNotFoundError(
            f"COCO dataset manifest not found: {args.manifest}. Run "
            "`python3 assets/coco/prepare_coco.py` from the repository root."
        )
    build_engine(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
