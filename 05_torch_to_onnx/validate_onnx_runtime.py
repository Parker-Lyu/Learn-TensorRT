#!/usr/bin/env python3
"""Compare YOLOv8n PyTorch raw output with ONNX Runtime output."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "assets" / "img.jpeg"
DEFAULT_WEIGHTS = REPO_ROOT / "assets" / "yolov8n.pt"
DEFAULT_ONNX = REPO_ROOT / "05_torch_to_onnx" / "outputs" / "yolov8n.onnx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "05_torch_to_onnx" / "outputs"


@dataclass(frozen=True)
class LetterboxInfo:
    original_width: int
    original_height: int
    input_width: int
    input_height: int
    resized_width: int
    resized_height: int
    scale: float
    pad_left: int
    pad_top: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLOv8n ONNX output against PyTorch.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="PyTorch .pt weights.")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="ONNX model path.")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Input image path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Square model input size.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance.")
    return parser.parse_args()


def letterbox_bgr(image: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxInfo]:
    if size <= 0:
        raise ValueError("--imgsz must be positive")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a BGR image with shape H x W x 3")

    original_height, original_width = image.shape[:2]
    scale = min(size / original_width, size / original_height)
    resized_width = int(round(original_width * scale))
    resized_height = int(round(original_height * scale))
    pad_left = (size - resized_width) // 2
    pad_top = (size - resized_height) // 2

    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized

    info = LetterboxInfo(
        original_width=original_width,
        original_height=original_height,
        input_width=size,
        input_height=size,
        resized_width=resized_width,
        resized_height=resized_height,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )
    return canvas, info


def preprocess_image(image_path: Path, size: int) -> tuple[np.ndarray, LetterboxInfo]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {image_path}")

    letterboxed, info = letterbox_bgr(image, size)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    nchw = np.ascontiguousarray(chw[None, ...], dtype=np.float32)
    return nchw, info


def run_pytorch(weights_path: Path, input_tensor: np.ndarray) -> np.ndarray:
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")

    model = YOLO(str(weights_path)).model.eval()
    tensor = torch.from_numpy(input_tensor)
    with torch.inference_mode():
        output = model(tensor)

    if isinstance(output, (tuple, list)):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"unexpected PyTorch output type: {type(output)!r}")
    return output.detach().cpu().numpy()


def run_onnxruntime(model_path: Path, input_tensor: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    outputs = session.run([output_meta.name], {input_meta.name: input_tensor})
    return outputs[0], {
        "input_name": input_meta.name,
        "input_shape": input_meta.shape,
        "output_name": output_meta.name,
        "output_shape": output_meta.shape,
        "providers": session.get_providers(),
    }


def compare_outputs(reference: np.ndarray, candidate: np.ndarray, rtol: float, atol: float) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: PyTorch {reference.shape}, ONNX Runtime {candidate.shape}")

    diff = np.abs(reference - candidate)
    max_index = np.unravel_index(int(np.argmax(diff)), diff.shape)
    close = np.isclose(reference, candidate, rtol=rtol, atol=atol)
    return {
        "shape": list(reference.shape),
        "dtype_reference": str(reference.dtype),
        "dtype_candidate": str(candidate.dtype),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "p99_abs_error": float(np.percentile(diff, 99)),
        "max_error_index": [int(v) for v in max_index],
        "reference_at_max_error": float(reference[max_index]),
        "candidate_at_max_error": float(candidate[max_index]),
        "rtol": rtol,
        "atol": atol,
        "allclose": bool(np.allclose(reference, candidate, rtol=rtol, atol=atol)),
        "close_fraction": float(close.mean()),
    }


def write_preview(path: Path, input_tensor: np.ndarray, torch_output: np.ndarray, ort_output: np.ndarray) -> None:
    lines = [
        f"input shape: {input_tensor.shape}",
        f"input dtype: {input_tensor.dtype}",
        "first 12 input values:",
        " ".join(f"{value:.6f}" for value in input_tensor.reshape(-1)[:12]),
        f"torch output shape: {torch_output.shape}",
        f"onnxruntime output shape: {ort_output.shape}",
        "first 12 torch output values:",
        " ".join(f"{value:.6f}" for value in torch_output.reshape(-1)[:12]),
        "first 12 onnxruntime output values:",
        " ".join(f"{value:.6f}" for value in ort_output.reshape(-1)[:12]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_tensor, letterbox = preprocess_image(args.image.resolve(), args.imgsz)
    torch_output = run_pytorch(args.weights.resolve(), input_tensor)
    ort_output, runtime_info = run_onnxruntime(args.onnx.resolve(), input_tensor)
    comparison = compare_outputs(torch_output, ort_output, rtol=args.rtol, atol=args.atol)

    np.save(output_dir / "input_nchw_float32.npy", input_tensor)
    np.save(output_dir / "pytorch_raw_output.npy", torch_output)
    np.save(output_dir / "onnxruntime_raw_output.npy", ort_output)
    write_preview(output_dir / "validation_preview.txt", input_tensor, torch_output, ort_output)

    report = {
        "weights": str(args.weights.resolve()),
        "onnx": str(args.onnx.resolve()),
        "image": str(args.image.resolve()),
        "letterbox": asdict(letterbox),
        "onnxruntime": runtime_info,
        "comparison": comparison,
    }
    report_path = output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"input: {input_tensor.shape} {input_tensor.dtype}")
    print(f"pytorch: {torch_output.shape} {torch_output.dtype}")
    print(f"onnxruntime: {ort_output.shape} {ort_output.dtype}")
    print(f"max abs error: {comparison['max_abs_error']:.8f}")
    print(f"mean abs error: {comparison['mean_abs_error']:.8f}")
    print(f"allclose(rtol={args.rtol}, atol={args.atol}): {comparison['allclose']}")
    print(f"report: {report_path}")

    if not comparison["allclose"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
