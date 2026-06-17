#!/usr/bin/env python3
"""Inspect ONNX tensor names, shapes, opset, and operator inventory."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = REPO_ROOT / "05_torch_to_onnx" / "outputs" / "yolov8n.onnx"
DEFAULT_REPORT = REPO_ROOT / "05_torch_to_onnx" / "outputs" / "onnx_inspection.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an ONNX model.")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="ONNX model path.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="JSON report path.")
    return parser.parse_args()


def dim_to_value(dim: onnx.TensorShapeProto.Dimension) -> str | int:
    if dim.dim_value:
        return int(dim.dim_value)
    if dim.dim_param:
        return dim.dim_param
    return "?"


def tensor_info(value_info: onnx.ValueInfoProto) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    dims = [dim_to_value(dim) for dim in tensor_type.shape.dim]
    elem_type = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
    return {"name": value_info.name, "dtype": elem_type, "shape": dims}


def inspect_model(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)

    ops = Counter(node.op_type for node in model.graph.node)
    opsets = {domain or "ai.onnx": version for domain, version in [(op.domain, op.version) for op in model.opset_import]}

    return {
        "model_path": str(model_path),
        "ir_version": model.ir_version,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "opsets": opsets,
        "inputs": [tensor_info(value) for value in model.graph.input],
        "outputs": [tensor_info(value) for value in model.graph.output],
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "operator_histogram": dict(sorted(ops.items())),
    }


def main() -> int:
    args = parse_args()
    report = inspect_model(args.onnx.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"checked: {report['model_path']}")
    print("inputs:")
    for item in report["inputs"]:
        print(f"  {item['name']}: {item['dtype']} {item['shape']}")
    print("outputs:")
    for item in report["outputs"]:
        print(f"  {item['name']}: {item['dtype']} {item['shape']}")
    print(f"nodes: {report['node_count']}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
