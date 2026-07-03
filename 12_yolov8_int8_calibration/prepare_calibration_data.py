#!/usr/bin/env python3
"""Create a tiny local calibration/validation image set for lesson 12 smoke tests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


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
    rotated = cv2.warpAffine(image, rotated_matrix, (image.shape[1], image.shape[0]), borderValue=(114, 114, 114))
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
    manifest.write_text(json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("../assets/dog.webp"))
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = read_image(args.source)
    calibration_dir = args.output_root / "calibration_smoke"
    validation_dir = args.output_root / "validation_smoke"
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

    print(f"Calibration images: {calibration_dir}")
    print(f"Validation images: {validation_dir}")
    print("This generated set is for smoke testing only; use representative real images for real INT8 decisions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
