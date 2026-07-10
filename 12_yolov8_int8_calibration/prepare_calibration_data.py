#!/usr/bin/env python3
"""Create a tiny local calibration/validation image set for lesson 12 smoke tests."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from dataset_manifest import build_manifest


@dataclass
class ImageRecord:
    path: str
    split: str
    transform: str


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return image


def transforms(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    darker = cv2.convertScaleAbs(image, alpha=0.75, beta=0)
    brighter = cv2.convertScaleAbs(image, alpha=1.15, beta=10)
    flipped = cv2.flip(image, 1)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    rotated_matrix = cv2.getRotationMatrix2D((image.shape[1] * 0.5, image.shape[0] * 0.5), 4.0, 1.0)
    rotated = cv2.warpAffine(
        image, rotated_matrix, (image.shape[1], image.shape[0]),
        borderValue=(114, 114, 114)
    )
    cropped = image[20:max(21, image.shape[0] - 20), 10:max(11, image.shape[1] - 10)]
    cropped = cv2.resize(cropped, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    return [
        ("original", image),
        ("darker", darker),
        ("brighter", brighter),
        ("flipped", flipped),
        ("blurred", blurred),
        ("rotated", rotated),
        ("cropped_resized", cropped),
    ]


def write_split(records: list[ImageRecord], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "manifest.json"
    manifest.write_text(
        json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../assets/dog.webp"))
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    parser.add_argument("--weights", type=Path, default=Path("../assets/yolov8n.pt"))
    return parser.parse_args()


def write_pseudo_labels(images: list[Path], labels_dir: Path, weights: Path) -> None:
    """Create smoke-only labels so the evaluator can exercise its metric path."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    labels_dir.mkdir(parents=True, exist_ok=True)
    for image_path in images:
        result = model.predict(
            source=str(image_path), conf=0.001, iou=0.7, max_det=300, verbose=False
        )[0]
        lines = []
        if result.boxes is not None:
            classes = result.boxes.cls.cpu().numpy()
            normalized_boxes = result.boxes.xywhn.cpu().numpy()
            for class_id, xywhn in zip(classes, normalized_boxes):
                coordinates = " ".join(f"{float(value):.8f}" for value in xywhn)
                lines.append(f"{int(class_id)} {coordinates}")
        (labels_dir / image_path.with_suffix(".txt").name).write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )


def main() -> int:
    args = parse_args()
    source = read_image(args.source)
    calibration_dir = args.output_root / "calibration_smoke"
    validation_dir = args.output_root / "validation_smoke"
    labels_dir = args.output_root / "validation_smoke_labels"
    for generated_dir in (calibration_dir, validation_dir, labels_dir):
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)

    records: list[ImageRecord] = []
    for index, (name, image) in enumerate(transforms(source)):
        split = "validation" if name in {"original", "brighter"} else "calibration"
        directory = validation_dir if split == "validation" else calibration_dir
        path = directory / f"{index:02d}_{name}.jpg"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"failed to write image: {path}")
        records.append(ImageRecord(path=str(path), split=split, transform=name))

    write_split([record for record in records if record.split == "calibration"], calibration_dir)
    write_split([record for record in records if record.split == "validation"], validation_dir)
    (args.output_root / "manifest.json").write_text(
        json.dumps([asdict(record) for record in records], indent=2),
        encoding="utf-8",
    )

    validation_images = sorted(validation_dir.glob("*.jpg"))
    write_pseudo_labels(validation_images, labels_dir, args.weights)
    dataset_manifest_path = args.output_root / "dataset_manifest.json"
    dataset_manifest = build_manifest(
        calibration_dir,
        validation_dir,
        labels_dir,
        dataset_manifest_path,
        "lesson12-generated-smoke-pseudolabels-v1",
    )
    dataset_manifest_path.write_text(json.dumps(dataset_manifest, indent=2), encoding="utf-8")

    print(f"Calibration images: {calibration_dir}")
    print(f"Validation images: {validation_dir}")
    print(f"Dataset manifest: {dataset_manifest_path}")
    print(
        "Validation labels are PyTorch pseudo-labels. This set tests plumbing only; "
        "never use its metrics for an INT8 release decision."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
