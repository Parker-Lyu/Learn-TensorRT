#!/usr/bin/env python3
"""Quantize YOLOv8n with ModelOpt PTQ and export an explicit Q/DQ ONNX model."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import modelopt
import modelopt.torch.quantization as mtq
import numpy as np
import onnx
import tensorrt as trt
import torch
import ultralytics
from ultralytics import YOLO

LESSON_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = LESSON_DIR.parent
sys.path.insert(0, str(LESSON_DIR))

from build_int8_engine import preprocess
from dataset_manifest import load_manifest, resolve_path, sha256


DEFAULT_WEIGHTS = REPO_ROOT / "assets/yolov8n.pt"
DEFAULT_MANIFEST = (
    LESSON_DIR
    / "outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json"
)
DEFAULT_OUTPUT_DIR = LESSON_DIR / "outputs/precision_recovery/05_modelopt_ptq"
INPUT_SHAPE = (1, 3, 640, 640)
PREPROCESS_ID = "letterbox114-bgr2rgb-fp32-div255-nchw-v1"
HIGH_PRECISION_DTYPES = {"fp32": "Float", "fp16": "Half"}


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def calibration_records(manifest_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None and limit <= 0:
        raise ValueError("calibration image limit must be positive")
    document = load_manifest(manifest_path)
    records = [record for record in document["records"] if record["split"] == "calibration"]
    if not records:
        raise ValueError("manifest contains no calibration records")
    if limit is not None:
        if limit > len(records):
            raise ValueError(
                f"requested {limit} calibration images, but manifest contains {len(records)}"
            )
        records = records[:limit]
    return records


def calibration_paths(manifest_path: Path, records: Sequence[dict[str, Any]]) -> list[Path]:
    paths = [resolve_path(manifest_path, str(record["image"])) for record in records]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"calibration image does not exist: {missing[0]}")
    return paths


class CalibrationBatches:
    """Read and preprocess calibration images one CPU batch at a time."""

    def __init__(
        self,
        paths: Sequence[Path],
        batch_size: int,
        input_shape: tuple[int, int, int, int] = INPUT_SHAPE,
    ) -> None:
        if not paths:
            raise ValueError("at least one calibration image is required")
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        if input_shape[0] != 1 or input_shape[1] != 3:
            raise ValueError(f"expected a 1x3xHxW input shape, got {input_shape}")
        self._paths = list(paths)
        self._batch_size = batch_size
        self._input_shape = input_shape

    def __len__(self) -> int:
        return (len(self._paths) + self._batch_size - 1) // self._batch_size

    @property
    def image_count(self) -> int:
        return len(self._paths)

    def __iter__(self) -> Iterator[np.ndarray]:
        for start in range(0, len(self._paths), self._batch_size):
            tensors = [
                preprocess(path, self._input_shape)
                for path in self._paths[start : start + self._batch_size]
            ]
            yield np.ascontiguousarray(np.concatenate(tensors, axis=0), dtype=np.float32)


class RawDetectionOutput(torch.nn.Module):
    """Expose only YOLO's raw [N, 84, 8400] prediction tensor."""

    def __init__(self, model: torch.nn.Module, internal_dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.model = model
        self.internal_dtype = internal_dtype

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.internal_dtype is not None:
            images = images.to(dtype=self.internal_dtype)
        output = self.model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError(f"unexpected YOLO output type: {type(output)!r}")
        return output.float()


def quantization_config(high_precision: str) -> tuple[dict[str, Any], str]:
    try:
        modelopt_dtype = HIGH_PRECISION_DTYPES[high_precision]
    except KeyError as error:
        raise ValueError(f"unsupported high-precision type: {high_precision}") from error
    config = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
    if config.get("algorithm") != "max":
        raise RuntimeError("the predeclared ModelOpt configuration must use max calibration")
    for pattern in ("*weight_quantizer", "*input_quantizer"):
        config["quant_cfg"][pattern]["trt_high_precision_dtype"] = modelopt_dtype
    return config, f"modelopt-int8-default-max-high-precision-{high_precision}-v1"


def calibrate_model(
    model: torch.nn.Module,
    batches: CalibrationBatches,
    device: torch.device,
    config: dict[str, Any],
) -> torch.nn.Module:
    def forward_loop(quantized_model: torch.nn.Module) -> None:
        completed = 0
        total = batches.image_count
        with torch.inference_mode():
            for batch in batches:
                inputs = torch.from_numpy(batch).to(device=device, non_blocking=True)
                quantized_model(inputs)
                completed += batch.shape[0]
                if completed == total or completed % 100 == 0:
                    print(f"Calibrated images: {completed}/{total}", flush=True)

    return mtq.quantize(model, config, forward_loop)


def export_qdq_onnx(
    model: torch.nn.Module,
    output_path: Path,
    device: torch.device,
    high_precision: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32, device=device)
    internal_dtype = None
    if high_precision == "fp16":
        model.half()
        internal_dtype = torch.float16
    elif high_precision != "fp32":
        raise ValueError(f"unsupported high-precision type: {high_precision}")
    wrapper = RawDetectionOutput(model, internal_dtype).eval()
    with torch.inference_mode():
        output = wrapper(dummy)
    if tuple(output.shape) != (1, 84, 8400):
        raise ValueError(f"unexpected YOLO output shape before export: {tuple(output.shape)}")

    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["output0"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def tensor_info(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    shape = []
    for dim in tensor_type.shape.dim:
        shape.append(int(dim.dim_value) if dim.dim_value else dim.dim_param or "?")
    return {
        "name": value_info.name,
        "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": shape,
    }


def inspect_qdq_onnx(path: Path, high_precision: str) -> dict[str, Any]:
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    operators = Counter(node.op_type for node in model.graph.node)
    quantize_count = operators["QuantizeLinear"]
    dequantize_count = operators["DequantizeLinear"]
    if quantize_count == 0 or dequantize_count == 0:
        raise ValueError("exported ONNX graph does not contain explicit Q/DQ nodes")

    inputs = [tensor_info(value) for value in model.graph.input]
    outputs = [tensor_info(value) for value in model.graph.output]
    expected_inputs = [{"name": "images", "dtype": "FLOAT", "shape": [1, 3, 640, 640]}]
    expected_outputs = [{"name": "output0", "dtype": "FLOAT", "shape": [1, 84, 8400]}]
    if inputs != expected_inputs or outputs != expected_outputs:
        raise ValueError(f"unexpected ONNX I/O contract: inputs={inputs}, outputs={outputs}")
    constant_dtypes = Counter()
    constant_values: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                constant_dtypes[onnx.TensorProto.DataType.Name(attribute.t.data_type)] += 1
                if node.op_type == "Constant" and node.output:
                    constant_values[node.output[0]] = onnx.numpy_helper.to_array(attribute.t)
    scale_audit = {
        "qdq_nodes_checked": 0,
        "nodes_with_nonpositive_scale": 0,
        "nodes_with_positive_subnormal_scale": 0,
    }
    for node in model.graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        scale = constant_values.get(node.input[1])
        if scale is None:
            continue
        scale_audit["qdq_nodes_checked"] += 1
        if np.any(scale <= 0):
            scale_audit["nodes_with_nonpositive_scale"] += 1
        if np.issubdtype(scale.dtype, np.floating):
            tiny = np.finfo(scale.dtype).tiny
            if np.any((scale > 0) & (scale < tiny)):
                scale_audit["nodes_with_positive_subnormal_scale"] += 1
    if scale_audit["nodes_with_nonpositive_scale"]:
        raise ValueError("Q/DQ graph contains non-positive scale coefficients")
    if high_precision == "fp16":
        if operators["Cast"] < 2:
            raise ValueError("FP16 high-precision graph is missing FP32 boundary casts")
        if constant_dtypes["FLOAT16"] == 0:
            raise ValueError("FP16 high-precision graph contains no FLOAT16 constants")
    elif high_precision != "fp32":
        raise ValueError(f"unsupported high-precision type: {high_precision}")
    return {
        "checker_passed": True,
        "inputs": inputs,
        "outputs": outputs,
        "node_count": len(model.graph.node),
        "quantize_linear_nodes": quantize_count,
        "dequantize_linear_nodes": dequantize_count,
        "qdq_scale_audit": scale_audit,
        "constant_tensor_dtype_histogram": dict(sorted(constant_dtypes.items())),
        "operator_histogram": dict(sorted(operators.items())),
    }


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "modelopt": modelopt.__version__,
        "tensorrt": trt.__version__,
        "onnx": onnx.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }


def write_metadata(
    path: Path,
    weights: Path,
    manifest: Path,
    records: Sequence[dict[str, Any]],
    onnx_path: Path,
    inspection: dict[str, Any],
    batch_size: int,
    candidate_kind: str,
    high_precision: str,
    config_id: str,
    config: dict[str, Any],
) -> None:
    document = load_manifest(manifest, verify_hashes=False)
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_kind": candidate_kind,
        "valid_for_accuracy_gate": candidate_kind == "formal",
        "weights": str(weights.resolve()),
        "weights_sha256": sha256(weights),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": sha256(manifest),
        "dataset_id": document["dataset_id"],
        "calibration_images": len(records),
        "calibration_image_sha256": [record["image_sha256"] for record in records],
        "calibration_batch_size": batch_size,
        "input_shape": list(INPUT_SHAPE),
        "preprocess": PREPROCESS_ID,
        "high_precision": high_precision,
        "quantization_config": config_id,
        "modelopt_config": config,
        "onnx": str(onnx_path.resolve()),
        "onnx_sha256": sha256(onnx_path),
        "onnx_inspection": inspection,
        "versions": package_versions(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--calibration-images", type=positive_int, default=3000)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument(
        "--candidate-kind", choices=("smoke", "formal"), default="formal",
        help="Smoke artifacts validate the pipeline but are never valid accuracy candidates.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="yolov8n_modelopt_int8_max_train3000")
    parser.add_argument(
        "--high-precision", choices=tuple(HIGH_PRECISION_DTYPES), default="fp32",
        help="Data type used by Q/DQ high-precision tensors inside the exported graph.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = args.weights.resolve()
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"weights file not found: {weights}")
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest file not found: {manifest}")
    if args.candidate_kind == "smoke" and not 32 <= args.calibration_images <= 64:
        raise ValueError("smoke calibration must use between 32 and 64 images")
    if args.candidate_kind == "formal" and args.calibration_images != 3000:
        raise ValueError("the primary formal candidate must use exactly 3,000 calibration images")
    if not torch.cuda.is_available():
        raise RuntimeError("ModelOpt calibration requires a CUDA device in this lesson")

    records = calibration_records(manifest, args.calibration_images)
    paths = calibration_paths(manifest, records)
    batches = CalibrationBatches(paths, args.batch_size)
    device = torch.device("cuda:0")
    model = YOLO(str(weights)).model.eval().to(device)
    config, config_id = quantization_config(args.high_precision)
    quantized_model = calibrate_model(model, batches, device, config)

    onnx_path = output_dir / f"{args.name}.onnx"
    metadata_path = output_dir / f"{args.name}.onnx.json"
    export_qdq_onnx(quantized_model, onnx_path, device, args.high_precision)
    inspection = inspect_qdq_onnx(onnx_path, args.high_precision)
    write_metadata(
        metadata_path, weights, manifest, records, onnx_path, inspection,
        args.batch_size, args.candidate_kind, args.high_precision, config_id, config,
    )
    print(f"ONNX: {onnx_path}")
    print(f"Metadata: {metadata_path}")
    print(
        "Q/DQ nodes: "
        f"{inspection['quantize_linear_nodes']}/{inspection['dequantize_linear_nodes']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
