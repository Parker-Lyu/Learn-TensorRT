#!/usr/bin/env python3
"""Create and validate explicit mixed-precision ONNX models with ModelOpt AutoCast."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort


REPO_ROOT = Path(__file__).resolve().parents[1]
LESSON_DIR = REPO_ROOT / "06_trtexec_engine"
DEFAULT_OUTPUT_DIR = LESSON_DIR / "outputs"
DEFAULT_REFERENCE_INPUT = REPO_ROOT / "05_torch_to_onnx/outputs/input_nchw_float32.npy"
DEFAULT_STATIC_ONNX = REPO_ROOT / "05_torch_to_onnx/outputs/yolov8n.onnx"
DEFAULT_DYNAMIC_ONNX = REPO_ROOT / "05_torch_to_onnx/outputs/yolov8n_dynamic.onnx"
DETECTION_HEAD_PATTERN = r"/model\.22/.*"


@dataclass(frozen=True)
class ConversionSpec:
    name: str
    source: Path
    output: Path
    representative_batch: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and validate strongly typed FP16-ready YOLOv8n ONNX graphs."
    )
    parser.add_argument("--static-onnx", type=Path, default=DEFAULT_STATIC_ONNX)
    parser.add_argument("--dynamic-onnx", type=Path, default=DEFAULT_DYNAMIC_ONNX)
    parser.add_argument("--reference-input", type=Path, default=DEFAULT_REFERENCE_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models", nargs="+", choices=("static", "dynamic"), default=("static", "dynamic")
    )
    parser.add_argument("--coordinate-rtol", type=float, default=5e-2)
    parser.add_argument("--coordinate-atol", type=float, default=5e-2)
    parser.add_argument("--score-rtol", type=float, default=1e-2)
    parser.add_argument("--score-atol", type=float, default=2e-2)
    parser.add_argument("--exclude-pattern", default=DETECTION_HEAD_PATTERN)
    return parser.parse_args()


def load_reference_input(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Validated lesson 05 input not found: {path}. "
            "Run python3 05_torch_to_onnx/validate_onnx_runtime.py first."
        )
    tensor = np.load(path, allow_pickle=False)
    if tensor.shape != (1, 3, 640, 640) or tensor.dtype != np.float32:
        raise ValueError(
            f"expected lesson 05 input float32 [1,3,640,640], got {tensor.dtype} {tensor.shape}"
        )
    if not np.isfinite(tensor).all():
        raise ValueError("reference input contains NaN or infinity")
    return tensor


def make_representative_input(tensor: np.ndarray, batch: int) -> np.ndarray:
    if tensor.shape != (1, 3, 640, 640) or tensor.dtype != np.float32:
        raise ValueError("representative input source must be float32 [1,3,640,640]")
    if batch <= 0:
        raise ValueError("representative batch must be positive")
    return np.repeat(tensor, batch, axis=0)


def tensor_shape(value: onnx.ValueInfoProto) -> list[int | str | None]:
    dimensions: list[int | str | None] = []
    for dimension in value.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(dimension.dim_value)
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return dimensions


def io_contract(model: onnx.ModelProto) -> dict[str, list[dict[str, Any]]]:
    def describe(values: Any) -> list[dict[str, Any]]:
        return [
            {
                "name": value.name,
                "element_type": value.type.tensor_type.elem_type,
                "shape": tensor_shape(value),
            }
            for value in values
        ]

    return {"inputs": describe(model.graph.input), "outputs": describe(model.graph.output)}


def compatible_io_contract(source: dict[str, list[dict[str, Any]]], candidate: dict[str, list[dict[str, Any]]]) -> bool:
    """Check callable ONNX I/O, without treating symbolic shape annotations as ABI."""
    for role in ("inputs", "outputs"):
        if len(source[role]) != len(candidate[role]):
            return False
        for before, after in zip(source[role], candidate[role]):
            if before["name"] != after["name"] or before["element_type"] != after["element_type"]:
                return False
            if len(before["shape"]) != len(after["shape"]):
                return False
    return True


def matching_node_count(model: onnx.ModelProto, pattern: str) -> int:
    compiled = re.compile(pattern)
    return sum(bool(compiled.search(node.name)) for node in model.graph.node)


def graph_precision_inventory(model: onnx.ModelProto) -> dict[str, int]:
    return {
        "fp32_initializers": sum(
            initializer.data_type == onnx.TensorProto.FLOAT
            for initializer in model.graph.initializer
        ),
        "fp16_initializers": sum(
            initializer.data_type == onnx.TensorProto.FLOAT16
            for initializer in model.graph.initializer
        ),
        "cast_nodes": sum(node.op_type == "Cast" for node in model.graph.node),
        "total_nodes": len(model.graph.node),
    }


def compare_outputs(
    reference_outputs: list[np.ndarray],
    candidate_outputs: list[np.ndarray],
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    if len(reference_outputs) != len(candidate_outputs):
        raise ValueError("reference and candidate output counts differ")

    reports: list[dict[str, Any]] = []
    passed = True
    for index, (reference, candidate) in enumerate(zip(reference_outputs, candidate_outputs)):
        if reference.shape != candidate.shape:
            raise ValueError(
                f"output {index} shape mismatch: {reference.shape} versus {candidate.shape}"
            )
        if reference.dtype != candidate.dtype:
            raise ValueError(
                f"output {index} dtype mismatch: {reference.dtype} versus {candidate.dtype}"
            )
        if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
            raise ValueError(f"output {index} contains NaN or infinity")

        absolute = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
        allowed = atol + rtol * np.abs(reference.astype(np.float64))
        close = absolute <= allowed
        output_passed = bool(np.all(close))
        passed = passed and output_passed
        worst_index = tuple(int(value) for value in np.unravel_index(np.argmax(absolute), absolute.shape))
        reports.append(
            {
                "index": index,
                "shape": list(reference.shape),
                "dtype": str(reference.dtype),
                "max_abs_error": float(np.max(absolute)),
                "mean_abs_error": float(np.mean(absolute)),
                "p99_abs_error": float(np.percentile(absolute, 99)),
                "close_fraction": float(np.mean(close)),
                "allclose": output_passed,
                "worst_absolute_error": {
                    "index": list(worst_index),
                    "reference": float(reference[worst_index]),
                    "candidate": float(candidate[worst_index]),
                    "absolute_error": float(absolute[worst_index]),
                    "relative_error": float(
                        absolute[worst_index] / max(abs(float(reference[worst_index])), 1e-12)
                    ),
                },
            }
        )

    return {"rtol": rtol, "atol": atol, "passed": passed, "outputs": reports}


def compare_yolov8_raw_outputs(
    reference_outputs: list[np.ndarray],
    candidate_outputs: list[np.ndarray],
    coordinate_rtol: float,
    coordinate_atol: float,
    score_rtol: float,
    score_atol: float,
) -> dict[str, Any]:
    """Validate YOLOv8 [N, 84, anchors] raw output with semantic tolerances.

    The first four channels contain pixel-scale box coordinates. Their FP16 rounding error
    must be assessed relatively. The remaining class-score channels are bounded probabilities
    and use a stricter tolerance. This is a conversion smoke gate, not mAP validation.
    """
    for values in (coordinate_rtol, coordinate_atol, score_rtol, score_atol):
        if values < 0:
            raise ValueError("validation tolerances must be non-negative")
    if len(reference_outputs) != 1 or len(candidate_outputs) != 1:
        raise ValueError("YOLOv8 conversion expects exactly one raw-output tensor")
    reference, candidate = reference_outputs[0], candidate_outputs[0]
    if reference.ndim != 3 or reference.shape[1] < 5:
        raise ValueError(f"expected YOLOv8 raw output [N,84,anchors], got {reference.shape}")
    coordinate = compare_outputs(
        [reference[:, :4, :]], [candidate[:, :4, :]], coordinate_rtol, coordinate_atol
    )
    scores = compare_outputs(
        [reference[:, 4:, :]], [candidate[:, 4:, :]], score_rtol, score_atol
    )
    return {
        "contract": "YOLOv8 raw output: coordinate channels 0:4 and score channels 4:",
        "passed": coordinate["passed"] and scores["passed"],
        "coordinate_channels": coordinate,
        "score_channels": scores,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_onnx(path: Path, tensor: np.ndarray) -> tuple[list[str], list[np.ndarray]]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"expected one ONNX input, found {len(inputs)}")
    output_names = [output.name for output in session.get_outputs()]
    outputs = session.run(output_names, {inputs[0].name: tensor})
    return output_names, outputs


def run_autocast(spec: ConversionSpec, calibration_path: Path, log_path: Path, pattern: str) -> None:
    command = [
        sys.executable,
        "-m",
        "modelopt.onnx.autocast",
        "--onnx_path",
        str(spec.source),
        "--output_path",
        str(spec.output),
        "--low_precision_type",
        "fp16",
        "--keep_io_types",
        "--calibration_data",
        str(calibration_path),
        "--providers",
        "cpu",
        "--nodes_to_exclude",
        pattern,
        "--log_level",
        "INFO",
    ]
    completed = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"ModelOpt AutoCast failed for {spec.name}; see {log_path}")
    if not spec.output.is_file():
        raise RuntimeError(f"ModelOpt AutoCast did not create {spec.output}")


def conversion_specs(args: argparse.Namespace) -> dict[str, ConversionSpec]:
    output_dir = args.output_dir.resolve()
    return {
        "static": ConversionSpec(
            "static",
            args.static_onnx.resolve(),
            output_dir / "yolov8n_static_autocast_fp16.onnx",
            1,
        ),
        "dynamic": ConversionSpec(
            "dynamic",
            args.dynamic_onnx.resolve(),
            output_dir / "yolov8n_dynamic_autocast_fp16.onnx",
            2,
        ),
    }


def convert_and_validate(
    spec: ConversionSpec,
    reference_input: np.ndarray,
    output_dir: Path,
    pattern: str,
    coordinate_rtol: float,
    coordinate_atol: float,
    score_rtol: float,
    score_atol: float,
) -> Path:
    if not spec.source.is_file():
        raise FileNotFoundError(f"source ONNX not found: {spec.source}")

    source_model = onnx.load(spec.source)
    onnx.checker.check_model(source_model)
    source_contract = io_contract(source_model)
    excluded_nodes = matching_node_count(source_model, pattern)
    if excluded_nodes == 0:
        raise ValueError(
            f"detection-head exclusion pattern {pattern!r} matched no nodes in {spec.source}"
        )

    tensor = make_representative_input(reference_input, spec.representative_batch)
    calibration_path = output_dir / f"{spec.name}_autocast_input.npz"
    np.savez(calibration_path, images=tensor)
    log_path = output_dir / f"{spec.name}_autocast.log"
    run_autocast(spec, calibration_path, log_path, pattern)

    converted_model = onnx.load(spec.output)
    onnx.checker.check_model(converted_model)
    converted_contract = io_contract(converted_model)
    if not compatible_io_contract(source_contract, converted_contract):
        raise ValueError("AutoCast changed callable model input/output names, types, or ranks")
    inventory = graph_precision_inventory(converted_model)
    if inventory["fp16_initializers"] == 0 or inventory["cast_nodes"] == 0:
        raise ValueError("AutoCast output does not contain explicit FP16 weights and Cast nodes")

    source_names, source_outputs = run_onnx(spec.source, tensor)
    converted_names, converted_outputs = run_onnx(spec.output, tensor)
    if source_names != converted_names:
        raise ValueError(f"output names changed: {source_names} versus {converted_names}")
    comparison = compare_yolov8_raw_outputs(
        source_outputs,
        converted_outputs,
        coordinate_rtol,
        coordinate_atol,
        score_rtol,
        score_atol,
    )

    report = {
        "model": spec.name,
        "source_onnx": str(spec.source),
        "source_sha256": sha256(spec.source),
        "converted_onnx": str(spec.output),
        "converted_sha256": sha256(spec.output),
        "representative_input": str(calibration_path),
        "representative_shape": list(tensor.shape),
        "provider": "CPUExecutionProvider",
        "excluded_node_pattern": pattern,
        "excluded_source_nodes": excluded_nodes,
        "io_contract": converted_contract,
        "source_io_contract": source_contract,
        "precision_inventory": inventory,
        "comparison": comparison,
        "versions": {
            "modelopt": importlib.metadata.version("nvidia-modelopt"),
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "onnx_graphsurgeon": importlib.metadata.version("onnx-graphsurgeon"),
        },
        "scope": "single representative raw-output conversion gate, not dataset-level mAP validation",
    }
    report_path = output_dir / f"{spec.name}_fp16_onnx_validation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not comparison["passed"]:
        raise RuntimeError(f"FP16 ONNX validation failed; see {report_path}")
    return report_path


def main() -> int:
    args = parse_args()
    try:
        if min(args.coordinate_rtol, args.coordinate_atol, args.score_rtol, args.score_atol) < 0:
            raise ValueError("validation tolerances must be non-negative")
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        reference_input = load_reference_input(args.reference_input.resolve())
        specs = conversion_specs(args)
        for name in args.models:
            report = convert_and_validate(
                specs[name],
                reference_input,
                output_dir,
                args.exclude_pattern,
                args.coordinate_rtol,
                args.coordinate_atol,
                args.score_rtol,
                args.score_atol,
            )
            print(f"{name}: {specs[name].output}")
            print(f"validation: {report}")
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
