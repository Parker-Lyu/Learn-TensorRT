#!/usr/bin/env python3
"""Convert a saved NCHW tensor into Polygraphy input JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from polygraphy.json import save_json


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_NPY = REPO_ROOT / "05_torch_to_onnx" / "outputs" / "input_nchw_float32.npy"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "06a_polygraphy_precision_alignment" / "outputs" / "input_data.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Polygraphy input JSON from a lesson 05 tensor dump.")
    parser.add_argument("--input-npy", type=Path, default=DEFAULT_INPUT_NPY, help="Input NCHW float32 .npy file.")
    parser.add_argument("--input-name", default="images", help="Model input tensor name.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_JSON, help="Polygraphy JSON output path.")
    return parser.parse_args()


def validate_tensor(tensor: np.ndarray, input_npy: Path) -> np.ndarray:
    if tensor.ndim != 4:
        raise ValueError(f"expected a 4D NCHW tensor in {input_npy}, got shape {tensor.shape}")
    if tensor.dtype != np.float32:
        tensor = tensor.astype(np.float32)
    return np.ascontiguousarray(tensor)


def main() -> int:
    args = parse_args()
    try:
        input_npy = args.input_npy.resolve()
        if not input_npy.exists():
            raise FileNotFoundError(
                f"input tensor not found: {input_npy}\n"
                "Run lesson 05 first: python3 05_torch_to_onnx/validate_onnx_runtime.py"
            )
        if not args.input_name:
            raise ValueError("--input-name cannot be empty")

        tensor = validate_tensor(np.load(input_npy), input_npy)
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_json([{args.input_name: tensor}], output_path)

        print(f"input tensor: {input_npy}")
        print(f"shape: {list(tensor.shape)}")
        print(f"dtype: {tensor.dtype}")
        print(f"polygraphy input: {output_path}")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
