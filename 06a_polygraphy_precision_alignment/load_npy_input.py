#!/usr/bin/env python3
"""Polygraphy data loader that feeds the lesson 05 NCHW tensor from .npy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_NPY = REPO_ROOT / "05_torch_to_onnx" / "outputs" / "input_nchw_float32.npy"
DEFAULT_INPUT_NAME = "images"


def _input_path() -> Path:
    return Path(os.environ.get("POLYGRAPHY_INPUT_NPY", DEFAULT_INPUT_NPY)).resolve()


def _input_name() -> str:
    return os.environ.get("POLYGRAPHY_INPUT_NAME", DEFAULT_INPUT_NAME)


def _validate_tensor(tensor: np.ndarray, input_npy: Path) -> np.ndarray:
    if tensor.ndim != 4:
        raise ValueError(f"expected a 4D NCHW tensor in {input_npy}, got shape {tensor.shape}")
    if tensor.dtype != np.float32:
        tensor = tensor.astype(np.float32)
    return np.ascontiguousarray(tensor)


def load_data() -> Iterator[dict[str, np.ndarray]]:
    input_npy = _input_path()
    input_name = _input_name()
    if not input_name:
        raise ValueError("POLYGRAPHY_INPUT_NAME cannot be empty")
    if not input_npy.exists():
        raise FileNotFoundError(
            f"input tensor not found: {input_npy}\n"
            "Run lesson 05 first: python3 05_torch_to_onnx/validate_onnx_runtime.py"
        )

    yield {input_name: _validate_tensor(np.load(input_npy), input_npy)}
