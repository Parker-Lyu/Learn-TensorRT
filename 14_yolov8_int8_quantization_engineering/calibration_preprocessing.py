"""Canonical CPU preprocessing shared by calibration-data selection and engine building."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def letterbox(image_bgr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("letterbox expects a three-channel BGR image")
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError("letterbox size must be positive")
    scale = min(width / image_bgr.shape[1], height / image_bgr.shape[0])
    resized_width = max(1, min(width, int(round(image_bgr.shape[1] * scale))))
    resized_height = max(1, min(height, int(round(image_bgr.shape[0] * scale))))
    resized = cv2.resize(
        image_bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    output = np.full((height, width, 3), 114, dtype=np.uint8)
    pad_left = (width - resized_width) // 2
    pad_top = (height - resized_height) // 2
    output[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized
    return output


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return image


def preprocess_image(
    image_bgr: np.ndarray, input_shape: tuple[int, int, int, int]
) -> np.ndarray:
    batch, channels, height, width = input_shape
    if batch != 1 or channels != 3 or height <= 0 or width <= 0:
        raise ValueError("input shape must be positive single-image NCHW RGB")
    letterboxed = letterbox(image_bgr, (width, height))
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor)


def preprocess(path: Path, input_shape: tuple[int, int, int, int]) -> np.ndarray:
    return preprocess_image(read_image(path), input_shape)


def input_luma_mean(path: Path, input_shape: tuple[int, int, int, int]) -> float:
    """Measure luminance on the exact normalized tensor presented to TensorRT."""
    tensor = preprocess(path, input_shape)[0]
    red, green, blue = tensor
    return float(np.mean(0.299 * red + 0.587 * green + 0.114 * blue))
