#!/usr/bin/env python3
"""Download and verify the canonical COCO dataset declared by the committed manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
LESSON12_DIR = REPO_ROOT / "12_yolov8_int8_calibration"
sys.path.insert(0, str(LESSON12_DIR))

from dataset_manifest import load_manifest, resolve_path  # noqa: E402

CANONICAL_MANIFEST = SCRIPT_DIR / "data/dataset_manifest.json"
ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
TRAIN_IMAGE_URL = "http://images.cocodataset.org/train2017/{file_name}"
ARCHIVE_SHA256 = {
    "annotations_trainval2017.zip": (
        "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
    ),
    "val2017.zip": "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
}
EXPECTED_CALIBRATION_IMAGES = 1_000
EXPECTED_VAL_IMAGES = 5_000
EXPECTED_CATEGORIES = 80
USER_AGENT = "Learn-TensorRT-COCO-preparer/1.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")


def download_file(
    url: str, destination: Path, retries: int = 3, show_progress: bool = False
) -> None:
    """Download one file atomically, resuming a partial transfer when supported."""
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")

    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                resumed = offset > 0 and getattr(response, "status", None) == 206
                downloaded = offset if resumed else 0
                content_length = int(response.headers.get("Content-Length", "0"))
                total = downloaded + content_length if content_length else 0
                next_report = downloaded
                with partial.open("ab" if resumed else "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        downloaded += len(chunk)
                        if show_progress and downloaded >= next_report:
                            total_text = f"/{total / 2**20:.0f}" if total else ""
                            print(
                                f"  {destination.name}: {downloaded / 2**20:.0f}"
                                f"{total_text} MiB"
                            )
                            step = max(total // 10, 100 * 2**20) if total else 100 * 2**20
                            next_report = downloaded + step
            partial.replace(destination)
            return
        except (OSError, urllib.error.URLError) as error:
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {error}") from error
            time.sleep(attempt)


def verify_archive(path: Path) -> None:
    verify_sha256(path, ARCHIVE_SHA256[path.name])
    try:
        with zipfile.ZipFile(path) as archive:
            corrupt_member = archive.testzip()
    except zipfile.BadZipFile as error:
        raise RuntimeError(f"invalid ZIP archive: {path}") from error
    if corrupt_member is not None:
        raise RuntimeError(f"corrupt ZIP member {corrupt_member!r} in {path}")


def canonical_records(document: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if document.get("schema_version") != 1:
        raise ValueError("canonical manifest must use schema version 1")
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError("canonical manifest records must be a list")
    selected = [record for record in records if record.get("split") == split]
    expected = EXPECTED_CALIBRATION_IMAGES if split == "calibration" else EXPECTED_VAL_IMAGES
    if len(selected) != expected:
        raise ValueError(f"expected {expected} {split} records, found {len(selected)}")
    if len({record["image"] for record in selected}) != len(selected):
        raise ValueError(f"canonical manifest contains duplicate {split} image paths")
    return selected


def load_canonical_document(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"committed canonical manifest is missing: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    calibration = canonical_records(document, "calibration")
    validation = canonical_records(document, "validation")
    if document.get("calibration_count") != len(calibration):
        raise ValueError("canonical calibration_count does not match its records")
    if document.get("validation_count") != len(validation):
        raise ValueError("canonical validation_count does not match its records")
    return document


def validate_jpeg(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 4:
        raise RuntimeError(f"missing or empty JPEG: {path}")
    with path.open("rb") as stream:
        if stream.read(3) != b"\xff\xd8\xff":
            raise RuntimeError(f"file is not a JPEG image: {path}")


def checked_destination(manifest_path: Path, record: dict[str, Any], prefix: str) -> Path:
    destination = resolve_path(manifest_path, record["image"]).resolve()
    allowed_root = (manifest_path.parent / prefix).resolve()
    try:
        destination.relative_to(allowed_root)
    except ValueError as error:
        raise ValueError(f"manifest path escapes {allowed_root}: {destination}") from error
    return destination


def download_calibration_images(
    records: list[dict[str, Any]], manifest_path: Path, workers: int
) -> list[Path]:
    def fetch(record: dict[str, Any]) -> Path:
        destination = checked_destination(
            manifest_path, record, "calibration/images"
        )
        expected_hash = record["image_sha256"]
        if destination.is_file():
            try:
                validate_jpeg(destination)
                verify_sha256(destination, expected_hash)
                return destination
            except RuntimeError:
                destination.unlink()
        file_name = urllib.parse.quote(destination.name)
        download_file(TRAIN_IMAGE_URL.format(file_name=file_name), destination)
        validate_jpeg(destination)
        try:
            verify_sha256(destination, expected_hash)
        except RuntimeError:
            destination.unlink(missing_ok=True)
            raise
        return destination

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, record) for record in records]
        paths = []
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            paths.append(future.result())
            if completed % 100 == 0 or completed == len(futures):
                print(f"  calibration images: {completed}/{len(futures)}")
    return sorted(paths)


def extract_annotation(archive_path: Path, member: str, output_dir: Path) -> Path:
    destination = output_dir / Path(member).name
    if destination.is_file():
        return destination
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with zipfile.ZipFile(archive_path) as archive:
        try:
            source = archive.open(member)
        except KeyError as error:
            raise RuntimeError(f"{member} is missing from {archive_path}") from error
        with source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)
    temporary.replace(destination)
    return destination


def extract_validation_images(
    archive_path: Path,
    output_dir: Path,
    records: list[dict[str, Any]],
    manifest_path: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_hashes = {
        checked_destination(manifest_path, record, "validation/images"):
        record["image_sha256"]
        for record in records
    }
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            name for name in archive.namelist()
            if name.startswith("val2017/") and name.lower().endswith(".jpg")
        )
        if len(members) != EXPECTED_VAL_IMAGES:
            raise RuntimeError(
                f"expected {EXPECTED_VAL_IMAGES} val images, found {len(members)}"
            )
        for name in members:
            destination = (output_dir / name).resolve()
            if destination not in expected_hashes:
                raise RuntimeError(f"validation ZIP member is absent from manifest: {name}")
            if destination.is_file():
                try:
                    validate_jpeg(destination)
                    verify_sha256(destination, expected_hashes[destination])
                    continue
                except RuntimeError:
                    destination.unlink()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            with archive.open(name) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.replace(destination)
            validate_jpeg(destination)
            verify_sha256(destination, expected_hashes[destination])
    return sorted((output_dir / "val2017").glob("*.jpg"))


def category_mapping(categories: list[dict[str, Any]]) -> dict[int, int]:
    ordered = sorted(categories, key=lambda category: category["id"])
    if len(ordered) != EXPECTED_CATEGORIES:
        raise ValueError(f"expected {EXPECTED_CATEGORIES} categories, found {len(ordered)}")
    return {category["id"]: index for index, category in enumerate(ordered)}


def yolo_line(
    annotation: dict[str, Any], image: dict[str, Any], class_id: int
) -> str | None:
    width = float(image["width"])
    height = float(image["height"])
    x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
    x1 = min(max(x, 0.0), width)
    y1 = min(max(y, 0.0), height)
    x2 = min(max(x + box_width, 0.0), width)
    y2 = min(max(y + box_height, 0.0), height)
    if x2 <= x1 or y2 <= y1:
        return None
    values = (
        (x1 + x2) * 0.5 / width,
        (y1 + y2) * 0.5 / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )
    return f"{class_id} " + " ".join(f"{value:.8f}" for value in values)


def write_validation_labels(document: dict[str, Any], labels_dir: Path) -> int:
    images = {image["id"]: image for image in document.get("images", [])}
    if len(images) != EXPECTED_VAL_IMAGES:
        raise ValueError(f"expected {EXPECTED_VAL_IMAGES} val images, found {len(images)}")
    mapping = category_mapping(document.get("categories", []))
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in document.get("annotations", []):
        if not annotation.get("iscrowd", 0):
            annotations_by_image[annotation["image_id"]].append(annotation)

    labels_dir.mkdir(parents=True, exist_ok=True)
    box_count = 0
    for image_id, image in sorted(images.items()):
        lines = []
        for annotation in annotations_by_image[image_id]:
            line = yolo_line(annotation, image, mapping[annotation["category_id"]])
            if line is not None:
                lines.append(line)
        box_count += len(lines)
        label_path = labels_dir / Path(image["file_name"]).with_suffix(".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return box_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

    manifest_path = CANONICAL_MANIFEST.resolve()
    manifest = load_canonical_document(manifest_path)
    output_root = manifest_path.parent
    downloads_dir = output_root / "downloads"
    annotations_archive = downloads_dir / "annotations_trainval2017.zip"
    val_archive = downloads_dir / "val2017.zip"

    print(f"Canonical dataset: {manifest['dataset_id']}")
    print("Downloading pinned COCO archives (existing files are reused)...")
    download_file(ANNOTATIONS_URL, annotations_archive, show_progress=True)
    download_file(VAL_IMAGES_URL, val_archive, show_progress=True)
    verify_archive(annotations_archive)
    verify_archive(val_archive)

    print("Downloading the exact calibration images declared by the manifest...")
    calibration_records = canonical_records(manifest, "calibration")
    calibration_paths = download_calibration_images(
        calibration_records, manifest_path, args.workers
    )

    print("Extracting val2017 and recreating the manifest-declared YOLO labels...")
    validation_records = canonical_records(manifest, "validation")
    validation_paths = extract_validation_images(
        val_archive,
        output_root / "validation/images",
        validation_records,
        manifest_path,
    )
    val_annotations = extract_annotation(
        annotations_archive,
        "annotations/instances_val2017.json",
        output_root / "annotations",
    )
    with val_annotations.open("r", encoding="utf-8") as stream:
        val_document = json.load(stream)
    box_count = write_validation_labels(
        val_document, output_root / "validation/labels/val2017"
    )

    # This verifies every downloaded image and generated label against the committed hashes.
    load_manifest(manifest_path)
    summary = {
        "dataset_id": manifest["dataset_id"],
        "calibration_images": len(calibration_paths),
        "validation_images": len(validation_paths),
        "validation_boxes_excluding_crowd": box_count,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "archive_sha256": {
            annotations_archive.name: sha256(annotations_archive),
            val_archive.name: sha256(val_archive),
        },
    }
    (output_root / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("Canonical COCO data is ready for lesson 12.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
