#!/usr/bin/env python3
"""Prove exact calibration/evaluation preprocessing parity on manifest images."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
LESSON12 = REPO_ROOT / "12_yolov8_int8_calibration"
LESSON09 = REPO_ROOT / "09_yolov8_trt_python"
sys.path.insert(0, str(LESSON12))
sys.path.insert(0, str(LESSON09))

import build_int8_engine as calibration  # noqa: E402
import infer_yolov8_trt as evaluation  # noqa: E402
from dataset_manifest import DEFAULT_COCO_MANIFEST, load_manifest, resolve_path  # noqa: E402

DEFAULT_OUTPUT = LESSON12 / "outputs/precision_recovery/01_preprocessing_parity.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_shape(text: str) -> tuple[int, int, int, int]:
    try:
        shape = tuple(int(part) for part in text.lower().split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must look like 1x3x640x640") from error
    if len(shape) != 4 or shape[0] != 1 or shape[1] != 3 or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape must be a positive single-image NCHW RGB shape")
    return shape  # type: ignore[return-value]


def compare_image(path: Path, input_shape: tuple[int, int, int, int]) -> dict[str, Any]:
    """Run the two production paths independently and require byte-identical tensors."""
    calibration_tensor = calibration.preprocess(path, input_shape)
    image_bgr = evaluation.read_image(path)
    evaluation_tensor, letterbox = evaluation.preprocess(image_bgr, input_shape)

    expected_shape = tuple(input_shape)
    invariants = {
        "calibration_shape": tuple(calibration_tensor.shape) == expected_shape,
        "evaluation_shape": tuple(evaluation_tensor.shape) == expected_shape,
        "calibration_dtype_fp32": calibration_tensor.dtype == np.float32,
        "evaluation_dtype_fp32": evaluation_tensor.dtype == np.float32,
        "calibration_c_contiguous": bool(calibration_tensor.flags.c_contiguous),
        "evaluation_c_contiguous": bool(evaluation_tensor.flags.c_contiguous),
    }
    if not all(invariants.values()):
        failed = [name for name, passed in invariants.items() if not passed]
        raise AssertionError(f"preprocessing invariant failure for {path}: {failed}")

    byte_identical = calibration_tensor.tobytes() == evaluation_tensor.tobytes()
    if byte_identical:
        max_abs_difference = 0.0
        differing_values = 0
    else:
        difference = np.abs(calibration_tensor - evaluation_tensor)
        max_abs_difference = float(np.max(difference))
        differing_values = int(np.count_nonzero(calibration_tensor != evaluation_tensor))

    return {
        "byte_identical": byte_identical,
        "max_abs_difference": max_abs_difference,
        "differing_values": differing_values,
        "tensor_bytes": calibration_tensor.nbytes,
        "letterbox": {
            "resized_width": letterbox.resized_width,
            "resized_height": letterbox.resized_height,
            "pad_left": letterbox.pad_left,
            "pad_top": letterbox.pad_top,
            "pad_right": letterbox.pad_right,
            "pad_bottom": letterbox.pad_bottom,
        },
    }


def verify_manifest(
    manifest_path: Path,
    input_shape: tuple[int, int, int, int],
    limit: int,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    records = [record for record in manifest["records"] if record["split"] == "calibration"]
    if not records:
        raise ValueError("manifest contains no calibration images")
    if limit < 0:
        raise ValueError("limit cannot be negative")
    selected = records if limit == 0 else records[:limit]

    failures = []
    total_tensor_bytes = 0
    for index, record in enumerate(selected, start=1):
        image_path = resolve_path(manifest_path, record["image"])
        result = compare_image(image_path, input_shape)
        total_tensor_bytes += result["tensor_bytes"]
        if not result["byte_identical"]:
            failures.append({"image": record["image"], **result})
        if index % 100 == 0 or index == len(selected):
            print(f"Compared calibration images: {index}/{len(selected)}", flush=True)

    return {
        "schema_version": 1,
        "step": "01_preprocessing_parity",
        "status": "PASS" if not failures else "FAIL",
        "requirement": "byte-identical FP32 NCHW tensors",
        "dataset": {
            "dataset_id": manifest["dataset_id"],
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "calibration_images_available": len(records),
            "calibration_images_checked": len(selected),
        },
        "input_shape": list(input_shape),
        "implementations": {
            "calibration": {
                "path": str(Path(calibration.__file__).resolve()),
                "sha256": sha256(Path(calibration.__file__)),
                "call": "build_int8_engine.preprocess",
            },
            "evaluation": {
                "path": str(Path(evaluation.__file__).resolve()),
                "sha256": sha256(Path(evaluation.__file__)),
                "call": "infer_yolov8_trt.read_image + preprocess",
            },
        },
        "comparison": {
            "images_failed": len(failures),
            "total_tensor_bytes_compared": total_tensor_bytes,
            "failures": failures,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_COCO_MANIFEST)
    parser.add_argument("--input-shape", type=parse_shape, default=(1, 3, 640, 640))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Check the first N calibration records; zero checks the complete calibration split.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(f"dataset manifest not found: {args.manifest}")
    report = verify_manifest(args.manifest, args.input_shape, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Evidence: {args.output}")
    print(f"Preprocessing parity: {report['status']}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
