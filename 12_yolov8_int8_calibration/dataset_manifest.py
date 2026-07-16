#!/usr/bin/env python3
"""Create and validate disjoint calibration and labeled-validation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_COCO_MANIFEST = (
    Path(__file__).resolve().parents[1] / "assets/coco/data/dataset_manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(directory: Path) -> list[Path]:
    paths = sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise FileNotFoundError(f"no images found under {directory}")
    return paths


def portable_path(path: Path, manifest_path: Path) -> str:
    try:
        return str(path.resolve().relative_to(manifest_path.parent.resolve()))
    except ValueError:
        return str(path.resolve())


def find_label(image: Path, image_root: Path, label_root: Path) -> Path:
    relative = image.relative_to(image_root).with_suffix(".txt")
    label = label_root / relative
    if not label.is_file():
        raise FileNotFoundError(f"missing YOLO label for {image}: expected {label}")
    return label


def build_manifest(calibration_dir: Path,
                   validation_dir: Path,
                   labels_dir: Path,
                   output: Path,
                   dataset_id: str) -> dict:
    calibration = image_paths(calibration_dir)
    validation = image_paths(validation_dir)
    calibration_hashes = {sha256(path) for path in calibration}
    validation_hashes = {sha256(path) for path in validation}
    overlap = calibration_hashes & validation_hashes
    if overlap:
        raise ValueError(
            f"calibration and validation contain {len(overlap)} byte-identical image(s)"
        )

    records = []
    for split, paths in (("calibration", calibration), ("validation", validation)):
        for image in paths:
            record = {
                "split": split,
                "image": portable_path(image, output),
                "image_sha256": sha256(image),
            }
            if split == "validation":
                label = find_label(image, validation_dir, labels_dir)
                record.update({
                    "label": portable_path(label, output),
                    "label_sha256": sha256(label),
                })
            records.append(record)

    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "annotation_format": "yolo_xywh_normalized",
        "calibration_count": len(calibration),
        "validation_count": len(validation),
        "records": records,
    }


def resolve_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest_path.parent / path


def load_manifest(path: Path, verify_hashes: bool = True) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported manifest schema in {path}")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"manifest has no records: {path}")

    calibration_hashes: set[str] = set()
    validation_hashes: set[str] = set()
    for record in records:
        split = record.get("split")
        if split not in {"calibration", "validation"}:
            raise ValueError(f"invalid split in manifest: {split}")
        image = resolve_path(path, record["image"])
        if not image.is_file():
            raise FileNotFoundError(f"manifest image does not exist: {image}")
        actual_hash = sha256(image)
        if verify_hashes and actual_hash != record.get("image_sha256"):
            raise ValueError(f"image hash changed since manifest creation: {image}")
        (calibration_hashes if split == "calibration" else validation_hashes).add(actual_hash)
        if split == "validation":
            label = resolve_path(path, record.get("label", ""))
            if not label.is_file():
                raise FileNotFoundError(f"manifest label does not exist: {label}")
            if verify_hashes and sha256(label) != record.get("label_sha256"):
                raise ValueError(f"label hash changed since manifest creation: {label}")

    overlap = calibration_hashes & validation_hashes
    if overlap:
        raise ValueError(f"manifest splits overlap by {len(overlap)} image hash(es)")
    return document


def records_for_split(path: Path, split: str) -> list[dict]:
    document = load_manifest(path)
    return [record for record in document["records"] if record["split"] == split]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-id", required=True,
        help="Versioned name, e.g. coco-val2017-subset-v1."
    )
    parser.add_argument("--output", type=Path, default=Path("data/dataset_manifest.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(
        args.calibration_dir, args.validation_dir, args.labels_dir, args.output, args.dataset_id
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {args.output}")
    print(f"Calibration images: {manifest['calibration_count']}")
    print(f"Labeled validation images: {manifest['validation_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
