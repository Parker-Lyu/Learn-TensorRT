#!/usr/bin/env python3
"""Classify TensorRT Engine Inspector JSON by role and actual tensor precision."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LESSON_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = LESSON_DIR / "outputs/precision_recovery/06_trt10_native_fp16_qdq"
DEFAULT_LAYERS = {
    "tensorrt_fp32": OUTPUT_DIR / "references/yolov8n_trt10_fp32.layers.json",
    "tensorrt_fp16": OUTPUT_DIR / "references/yolov8n_trt10_fp16.layers.json",
    "tensorrt_int8": OUTPUT_DIR / "candidate/yolov8n_modelopt_hp_fp16_trt10.layers.json",
}
INFRASTRUCTURE_TYPES = {
    "constant", "copy", "identity", "noop", "reformat", "shape", "shuffle",
}
COMPUTE_HINTS = (
    "convolution", "pointwise", "elementwise", "pool", "resize", "softmax",
    "activation", "matrix", "reduce", "slice", "concatenation", "plugin",
)


def tensor_precision(tensor: dict[str, Any]) -> str:
    value = str(tensor.get("Format/Datatype", "")).lower()
    if "int8" in value:
        return "INT8"
    if "fp16" in value or "half" in value:
        return "FP16"
    if "fp32" in value or "float" in value:
        return "FP32"
    if "int32" in value:
        return "INT32"
    return "OTHER"


def layer_role(layer: dict[str, Any]) -> str:
    layer_type = str(layer.get("LayerType", "")).lower()
    parameter_type = str(layer.get("ParameterType", "")).lower()
    combined = f"{layer_type} {parameter_type}"
    if any(name in combined for name in INFRASTRUCTURE_TYPES):
        return "infrastructure"
    if any(hint in combined for hint in COMPUTE_HINTS):
        return "compute"
    return "compute_other"


def is_int8_weight_convolution(layer: dict[str, Any]) -> bool:
    parameter_type = str(layer.get("ParameterType", "")).lower()
    layer_type = str(layer.get("LayerType", "")).lower()
    weight_type = str(layer.get("Weights", {}).get("Type", "")).lower()
    return "convolution" in f"{parameter_type} {layer_type}" and "int8" in weight_type


def output_category(layer: dict[str, Any]) -> str:
    precisions = {tensor_precision(item) for item in layer.get("Outputs", [])}
    precisions.discard("OTHER")
    if not precisions:
        return "OTHER"
    if len(precisions) == 1:
        return next(iter(precisions))
    return "+".join(sorted(precisions))


def is_external_boundary_conversion(layer: dict[str, Any]) -> bool:
    if layer_role(layer) != "infrastructure" or "reformat" not in str(
        layer.get("LayerType", "")
    ).lower():
        return False
    tensors = [*layer.get("Inputs", []), *layer.get("Outputs", [])]
    names = {str(item.get("Name", "")) for item in tensors}
    precisions = {tensor_precision(item) for item in tensors}
    return bool(names & {"images", "output0"}) and "FP32" in precisions and len(precisions) > 1


def classify(data: dict[str, Any]) -> dict[str, Any]:
    layers = data.get("Layers")
    if not isinstance(layers, list):
        raise ValueError("Engine Inspector JSON must contain a Layers list")
    roles = Counter()
    types = Counter()
    compute_outputs = Counter()
    int8_weight_outputs = Counter()
    reformats = 0
    qdq_reformats = 0
    boundary_conversions = 0
    pure_fp16_compute = []
    pure_fp32_compute = []

    for layer in layers:
        role = layer_role(layer)
        roles[role] += 1
        types[str(layer.get("LayerType", "UNKNOWN"))] += 1
        if role.startswith("compute"):
            category = output_category(layer)
            compute_outputs[category] += 1
            if category == "FP16":
                pure_fp16_compute.append(str(layer.get("Name", "")))
            elif category == "FP32":
                pure_fp32_compute.append(str(layer.get("Name", "")))
            if is_int8_weight_convolution(layer):
                int8_weight_outputs[category] += 1
        if "reformat" in str(layer.get("LayerType", "")).lower():
            reformats += 1
            if str(layer.get("Origin", "")).upper() == "QDQ":
                qdq_reformats += 1
        if is_external_boundary_conversion(layer):
            boundary_conversions += 1

    return {
        "total_layers": len(layers),
        "role_counts": dict(sorted(roles.items())),
        "layer_type_counts": dict(sorted(types.items())),
        "compute_output_precision_counts": dict(sorted(compute_outputs.items())),
        "int8_weight_convolutions_by_output_precision": dict(sorted(int8_weight_outputs.items())),
        "pure_fp16_compute_count": len(pure_fp16_compute),
        "pure_fp16_compute_layers": pure_fp16_compute,
        "pure_fp32_compute_count": len(pure_fp32_compute),
        "pure_fp32_compute_layers": pure_fp32_compute,
        "reformat_count": reformats,
        "qdq_origin_reformat_count": qdq_reformats,
        "fp32_external_boundary_conversion_count": boundary_conversions,
    }


def inspect(path: Path) -> dict[str, Any]:
    return classify(json.loads(path.read_text(encoding="utf-8")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-layers", type=Path, default=DEFAULT_LAYERS["tensorrt_fp32"])
    parser.add_argument("--fp16-layers", type=Path, default=DEFAULT_LAYERS["tensorrt_fp16"])
    parser.add_argument("--int8-layers", type=Path, default=DEFAULT_LAYERS["tensorrt_int8"])
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "layer_audit.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        "tensorrt_fp32": args.fp32_layers,
        "tensorrt_fp16": args.fp16_layers,
        "tensorrt_int8": args.int8_layers,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Engine Inspector JSON: " + ", ".join(missing))
    report = {
        "schema_version": 1,
        "engines": {
            name: {"layer_info": str(path.resolve()), **inspect(path)}
            for name, path in paths.items()
        },
        "step05_baseline": {
            "total_layers": 171,
            "reformat_count": 67,
            "compute_output_precision_counts": {"INT8": 49, "FP16": 32, "FP32": 12},
            "non_int8_pure_fp32_compute_count": 5,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Layer audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
