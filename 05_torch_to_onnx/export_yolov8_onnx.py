#!/usr/bin/env python3
"""Export YOLOv8n weights to ONNX for later TensorRT lessons."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = REPO_ROOT / "assets" / "yolov8n.pt"
DEFAULT_OUTPUT = REPO_ROOT / "05_torch_to_onnx" / "outputs" / "yolov8n.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLOv8n PyTorch weights to ONNX.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path for YOLOv8n .pt weights. Missing default weights are downloaded.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination ONNX file.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Square export image size.")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--dynamic", action="store_true", help="Export dynamic batch/shape axes.")
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Ask Ultralytics to simplify the ONNX graph. Requires onnxslim or onnxsim.",
    )
    return parser.parse_args()


def ensure_weights(weights_path: Path) -> Path:
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    if weights_path.exists():
        return weights_path

    if weights_path.name == "yolov8n.pt":
        downloaded = Path(attempt_download_asset(str(weights_path)))
        if downloaded.exists():
            return downloaded

    raise FileNotFoundError(
        f"weights file not found: {weights_path}. "
        "Pass --weights or place yolov8n.pt under the root assets directory."
    )


def export_onnx(
    weights_path: Path,
    output_path: Path,
    image_size: int,
    opset: int,
    dynamic: bool,
    simplify: bool,
) -> Path:
    if image_size <= 0:
        raise ValueError("--imgsz must be positive")
    if opset <= 0:
        raise ValueError("--opset must be positive")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights_path))
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=image_size,
            opset=opset,
            dynamic=dynamic,
            simplify=simplify,
        )
    )

    if not exported_path.exists():
        raise RuntimeError(f"Ultralytics reported {exported_path}, but the file does not exist")

    if exported_path.resolve() != output_path.resolve():
        shutil.copy2(exported_path, output_path)
        if exported_path.parent == weights_path.parent:
            exported_path.unlink()

    return output_path


def main() -> int:
    args = parse_args()
    weights_path = ensure_weights(args.weights.resolve())
    output_path = export_onnx(
        weights_path=weights_path,
        output_path=args.output.resolve(),
        image_size=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        simplify=args.simplify,
    )

    print(f"weights: {weights_path}")
    print(f"onnx: {output_path}")
    print(f"dynamic: {args.dynamic}")
    print(f"simplify: {args.simplify}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
