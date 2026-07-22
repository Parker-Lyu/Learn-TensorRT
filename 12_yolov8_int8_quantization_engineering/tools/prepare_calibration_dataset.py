#!/usr/bin/env python3
"""Reproduce the fixed coverage-aware COCO calibration selection contract."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LESSON_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = REPO_ROOT / "assets/coco/data/downloads/annotations_trainval2017.zip"
DEFAULT_VALIDATION_MANIFEST = REPO_ROOT / "assets/coco/data/dataset_manifest.json"
DEFAULT_CANDIDATE_DIR = (
    REPO_ROOT / "assets/coco/data/calibration/images/train2017_quantization_v3_candidates_n5000"
)
DEFAULT_TARGET_DIR = (
    REPO_ROOT / "assets/coco/data/calibration/images/train2017_quantization_v3_n3000"
)
DEFAULT_MANIFEST = LESSON_DIR / "data/dataset_manifest.json"
DEFAULT_SELECTION_CONTRACT = LESSON_DIR / "data/calibration_selection.json"
DEFAULT_REGENERATED_SELECTION = (
    LESSON_DIR / "outputs/data_preparation/regenerated_calibration_selection.json"
)
DEFAULT_REPORT = LESSON_DIR / "outputs/data_preparation/coverage_report.json"
TRAIN_IMAGE_URL = "http://images.cocodataset.org/train2017/{file_name}"
USER_AGENT = "Learn-TensorRT-lesson12-data-preparer/1.0"
EXPECTED_CANDIDATE_COUNT = 5_000
EXPECTED_TARGET_COUNT = 3_000
EXPECTED_ALGORITHM = "category-seeded-farthest-coverage"
EXPECTED_ALGORITHM_VERSION = 3
FEATURE_NAMES = (
    "log1p_object_count",
    "small_ratio",
    "medium_ratio",
    "large_ratio",
    "log_image_aspect",
    "log_mean_relative_box_area",
    "luma_mean",
    "luma_std",
    "dark_pixel_fraction",
    "bright_pixel_fraction",
    "mean_saturation",
    "edge_fraction",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def load_selection_contract(path: Path) -> dict[str, Any]:
    """Load the committed candidate-pool identity and selected-image contract."""
    if not path.is_file():
        raise FileNotFoundError(f"selection contract not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("selection contract must use schema version 1")
    if document.get("algorithm") != EXPECTED_ALGORITHM:
        raise ValueError("selection contract declares an unexpected algorithm")
    if document.get("algorithm_version") != EXPECTED_ALGORITHM_VERSION:
        raise ValueError("selection contract declares an unexpected algorithm version")
    if not valid_sha256(document.get("annotations_sha256")):
        raise ValueError("selection contract has an invalid annotation SHA-256")

    records = document.get("candidate_pool")
    if not isinstance(records, list) or len(records) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError(
            f"selection contract must contain {EXPECTED_CANDIDATE_COUNT} candidates"
        )
    candidate_ids: set[int] = set()
    file_names: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("selection contract candidate records must be objects")
        current_id = record.get("image_id")
        file_name = record.get("file_name")
        if not isinstance(current_id, int) or current_id <= 0:
            raise ValueError("selection contract contains an invalid image ID")
        expected_name = f"{current_id:012d}.jpg"
        if file_name != expected_name:
            raise ValueError(
                f"selection contract file name mismatch for image {current_id}: {file_name}"
            )
        if not valid_sha256(record.get("image_sha256")):
            raise ValueError(f"selection contract has an invalid hash for image {current_id}")
        if current_id in candidate_ids or file_name in file_names:
            raise ValueError(f"selection contract contains duplicate image {current_id}")
        candidate_ids.add(current_id)
        file_names.add(file_name)

    selected_ids = document.get("selected_ids")
    if not isinstance(selected_ids, list) or len(selected_ids) != EXPECTED_TARGET_COUNT:
        raise ValueError(
            f"selection contract must contain {EXPECTED_TARGET_COUNT} selected IDs"
        )
    if len(set(selected_ids)) != len(selected_ids):
        raise ValueError("selection contract contains duplicate selected IDs")
    if any(not isinstance(current_id, int) for current_id in selected_ids):
        raise ValueError("selection contract contains a non-integer selected ID")
    missing = sorted(set(selected_ids) - candidate_ids)
    if missing:
        raise ValueError(f"selected IDs are absent from the candidate pool: {missing[:5]}")
    return document


def download_file(url: str, destination: Path, retries: int = 3) -> None:
    """Download one file atomically, retrying transient network failures."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open(
                "wb"
            ) as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            partial.replace(destination)
            return
        except (OSError, urllib.error.URLError) as error:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {error}") from error
            time.sleep(attempt)


def ensure_candidate_image(
    record: dict[str, Any],
    directory: Path,
    downloader: Callable[[str, Path], None] = download_file,
) -> Path:
    """Materialize one contract image and reject content that does not match its hash."""
    destination = directory / record["file_name"]
    expected_hash = record["image_sha256"]
    if destination.is_file() and sha256(destination) == expected_hash:
        return destination
    destination.unlink(missing_ok=True)
    file_name = urllib.parse.quote(record["file_name"])
    downloader(TRAIN_IMAGE_URL.format(file_name=file_name), destination)
    if not destination.is_file() or sha256(destination) != expected_hash:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded candidate hash mismatch: {record['file_name']}")
    return destination


def download_candidate_pool(
    contract: dict[str, Any], directory: Path, workers: int
) -> None:
    if workers <= 0:
        raise ValueError("--workers must be positive")
    directory.mkdir(parents=True, exist_ok=True)
    records = contract["candidate_pool"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(ensure_candidate_image, record, directory)
            for record in records
        ]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            future.result()
            if completed % 250 == 0 or completed == len(futures):
                print(f"  candidate images: {completed}/{len(futures)}", flush=True)


def image_id(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as error:
        raise ValueError(f"COCO image name must be a numeric image ID: {path}") from error


def load_train_annotations(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"COCO annotation archive not found: {path}")
    with zipfile.ZipFile(path) as archive:
        with archive.open("annotations/instances_train2017.json") as stream:
            return json.load(stream)


def size_name(area: float) -> str:
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def build_profiles(document: dict[str, Any]) -> dict[int, dict[str, Any]]:
    category_ids = sorted(int(category["id"]) for category in document["categories"])
    if len(category_ids) != 80:
        raise ValueError(f"expected 80 COCO categories, found {len(category_ids)}")
    class_index = {category_id: index for index, category_id in enumerate(category_ids)}
    profiles = {
        int(image["id"]): {
            "id": int(image["id"]),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "classes": set(),
            "class_counts": Counter(),
            "size_counts": Counter(),
            "box_areas": [],
        }
        for image in document["images"]
    }
    for annotation in document["annotations"]:
        if annotation.get("iscrowd", 0):
            continue
        profile = profiles[int(annotation["image_id"])]
        class_id = class_index[int(annotation["category_id"])]
        area = max(0.0, float(annotation.get("area", 0.0)))
        profile["classes"].add(class_id)
        profile["class_counts"][class_id] += 1
        profile["size_counts"][size_name(area)] += 1
        image_area = max(1.0, float(profile["width"] * profile["height"]))
        profile["box_areas"].append(area / image_area)
    for profile in profiles.values():
        profile["classes"] = tuple(sorted(profile["classes"]))
        profile["object_count"] = int(sum(profile["class_counts"].values()))
    return profiles


def collect_candidates(directories: Iterable[Path]) -> dict[int, Path]:
    candidates: dict[int, Path] = {}
    for directory in directories:
        if not directory.is_dir():
            raise FileNotFoundError(f"candidate image directory not found: {directory}")
        for path in sorted(directory.glob("*.jpg")):
            current_id = image_id(path)
            if current_id in candidates:
                raise ValueError(f"duplicate candidate image ID {current_id}: {path}")
            candidates[current_id] = path
    if not candidates:
        raise ValueError("candidate pool is empty")
    return candidates


def validate_candidate_pool(
    contract: dict[str, Any], candidates: dict[int, Path]
) -> dict[int, str]:
    expected = {record["image_id"]: record for record in contract["candidate_pool"]}
    missing = sorted(set(expected) - set(candidates))
    extra = sorted(set(candidates) - set(expected))
    if missing or extra:
        raise ValueError(
            "candidate pool does not match the fixed contract: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    hashes: dict[int, str] = {}
    for current_id in sorted(expected):
        path = candidates[current_id]
        record = expected[current_id]
        if path.name != record["file_name"]:
            raise ValueError(
                f"candidate file name mismatch for image {current_id}: {path.name}"
            )
        actual_hash = sha256(path)
        if actual_hash != record["image_sha256"]:
            raise ValueError(f"candidate image hash mismatch: {path}")
        hashes[current_id] = actual_hash
    return hashes


def require_selection_match(
    contract: dict[str, Any], regenerated: dict[str, Any]
) -> None:
    if regenerated == contract:
        return
    expected_ids = contract["selected_ids"]
    actual_ids = regenerated["selected_ids"]
    for index, (expected, actual) in enumerate(zip(expected_ids, actual_ids)):
        if expected != actual:
            raise RuntimeError(
                "selection contract mismatch: "
                f"position {index} expected image {expected}, computed image {actual}. "
                "Use the pinned course environment or create an explicit new dataset version; "
                "the committed contract was not modified."
            )
    raise RuntimeError(
        "selection contract metadata changed even though selected IDs match; "
        "the committed contract was not modified"
    )


def numeric_features(profile: dict[str, Any], path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read candidate image: {path}")
    height, width = image.shape[:2]
    scale = min(256.0 / max(width, 1), 256.0 / max(height, 1), 1.0)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny((gray * 255.0).astype(np.uint8), 80, 160)
    count = max(1, profile["object_count"])
    sizes = profile["size_counts"]
    box_areas = profile["box_areas"]
    return np.asarray(
        [
            np.log1p(profile["object_count"]),
            sizes["small"] / count,
            sizes["medium"] / count,
            sizes["large"] / count,
            np.log(max(profile["width"], 1) / max(profile["height"], 1)),
            np.log1p(10_000.0 * (float(np.mean(box_areas)) if box_areas else 0.0)),
            float(np.mean(gray)),
            float(np.std(gray)),
            float(np.mean(gray < 0.15)),
            float(np.mean(gray > 0.85)),
            float(np.mean(hsv[..., 1])) / 255.0,
            float(np.mean(edges > 0)),
        ],
        dtype=np.float64,
    )


def category_seeds(profiles: list[dict[str, Any]]) -> list[int]:
    """Greedily seed coverage without preserving any previous calibration membership."""
    uncovered = set(range(80))
    available = set(range(len(profiles)))
    selected: list[int] = []
    while uncovered:
        index = max(
            available,
            key=lambda candidate: (
                len(uncovered.intersection(profiles[candidate]["classes"])),
                len(profiles[candidate]["classes"]),
                sum(int(profiles[candidate]["size_counts"][name] > 0) for name in (
                    "small", "medium", "large"
                )),
                min(profiles[candidate]["object_count"], 50),
                -profiles[candidate]["id"],
            ),
        )
        newly_covered = uncovered.intersection(profiles[index]["classes"])
        if not newly_covered:
            raise ValueError(f"candidate pool does not cover COCO classes {sorted(uncovered)}")
        selected.append(index)
        available.remove(index)
        uncovered.difference_update(newly_covered)
    return selected


def robust_scale(features: np.ndarray) -> np.ndarray:
    low = np.percentile(features, 5, axis=0)
    high = np.percentile(features, 95, axis=0)
    return np.clip((features - low) / np.maximum(high - low, 1e-9), -0.5, 1.5)


def coverage_selection(
    profiles: list[dict[str, Any]], features: np.ndarray, target_count: int
) -> list[int]:
    if features.ndim != 2 or len(features) != len(profiles):
        raise ValueError("profiles and feature rows must have matching lengths")
    if target_count <= 0 or target_count > len(profiles):
        raise ValueError("target count must be within the candidate pool")
    selected = category_seeds(profiles)
    if len(selected) > target_count:
        raise ValueError("target count is too small to preserve category coverage")
    scaled = robust_scale(features)
    nearest = np.full(len(scaled), np.inf, dtype=np.float64)
    available = np.ones(len(scaled), dtype=bool)
    for index in selected:
        available[index] = False
        nearest = np.minimum(nearest, np.sum((scaled - scaled[index]) ** 2, axis=1))
    while len(selected) < target_count:
        scores = np.where(available, nearest, -1.0)
        index = int(np.argmax(scores))
        selected.append(index)
        available[index] = False
        nearest = np.minimum(nearest, np.sum((scaled - scaled[index]) ** 2, axis=1))
    return selected


def relative_path(target: Path, manifest: Path) -> str:
    return os.path.relpath(target.resolve(), manifest.parent.resolve())


def load_validation_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    records = [record for record in document["records"] if record["split"] == "validation"]
    if len(records) != int(document["validation_count"]):
        raise ValueError("validation manifest count does not match its records")
    return document, records


def source_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else manifest.parent / path


def summarize(profiles: list[dict[str, Any]], features: np.ndarray) -> dict[str, Any]:
    class_images = Counter(class_id for profile in profiles for class_id in profile["classes"])
    size_objects = Counter()
    for profile in profiles:
        size_objects.update(profile["size_counts"])
    return {
        "images": len(profiles),
        "classes_covered": sum(class_images[index] > 0 for index in range(80)),
        "class_image_occurrences": {str(index): class_images[index] for index in range(80)},
        "object_size_counts": {
            name: size_objects[name] for name in ("small", "medium", "large")
        },
        "feature_percentiles": {
            name: {
                "p05": float(np.percentile(features[:, index], 5)),
                "p50": float(np.percentile(features[:, index], 50)),
                "p95": float(np.percentile(features[:, index], 95)),
            }
            for index, name in enumerate(FEATURE_NAMES)
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        action="append",
        dest="candidate_dirs",
        help="Directory containing contract candidates; repeat for existing split pools.",
    )
    parser.add_argument(
        "--download-candidates",
        action="store_true",
        help="Download the exact fixed candidate pool before reproducing the selection.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--selection-contract",
        type=Path,
        default=DEFAULT_SELECTION_CONTRACT,
        help="Read-only committed identity for the candidate pool and selected IDs.",
    )
    parser.add_argument(
        "--regenerated-selection", type=Path, default=DEFAULT_REGENERATED_SELECTION
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--materialize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract = load_selection_contract(args.selection_contract)
    candidate_dirs = tuple(args.candidate_dirs or (DEFAULT_CANDIDATE_DIR,))
    if args.download_candidates:
        if len(candidate_dirs) != 1:
            raise ValueError("--download-candidates requires exactly one --candidate-dir")
        print("Downloading and verifying the fixed candidate pool...", flush=True)
        download_candidate_pool(contract, candidate_dirs[0], args.workers)
    candidates = collect_candidates(candidate_dirs)
    candidate_hashes = validate_candidate_pool(contract, candidates)

    annotation_hash = sha256(args.annotations) if args.annotations.is_file() else None
    if annotation_hash != contract["annotations_sha256"]:
        raise ValueError(
            "COCO annotation archive is missing or does not match the selection contract: "
            f"{args.annotations}"
        )
    annotations = load_train_annotations(args.annotations)
    profiles_by_id = build_profiles(annotations)
    del annotations
    missing = sorted(set(candidates) - profiles_by_id.keys())
    if missing:
        raise ValueError(f"candidate IDs are missing from train2017 annotations: {missing[:5]}")

    ordered_ids = sorted(candidates)
    profiles = [profiles_by_id[current_id] for current_id in ordered_ids]
    print(f"Computing coverage features for {len(ordered_ids)} candidate images...", flush=True)
    features = np.vstack(
        [numeric_features(profile, candidates[current_id]) for current_id, profile in zip(
            ordered_ids, profiles
        )]
    )
    selected_offsets = coverage_selection(profiles, features, len(contract["selected_ids"]))
    selected_ids = [ordered_ids[index] for index in selected_offsets]
    selected_profiles = [profiles[index] for index in selected_offsets]
    selected_features = features[selected_offsets]

    regenerated_selection = {
        "schema_version": 1,
        "algorithm": EXPECTED_ALGORITHM,
        "algorithm_version": EXPECTED_ALGORITHM_VERSION,
        "annotations_sha256": annotation_hash,
        "candidate_pool": [
            {
                "image_id": current_id,
                "file_name": candidates[current_id].name,
                "image_sha256": candidate_hashes[current_id],
            }
            for current_id in ordered_ids
        ],
        "selected_ids": selected_ids,
    }
    args.regenerated_selection.parent.mkdir(parents=True, exist_ok=True)
    args.regenerated_selection.write_text(
        json.dumps(regenerated_selection, indent=2) + "\n", encoding="utf-8"
    )
    require_selection_match(contract, regenerated_selection)
    print("Selection contract reproduced exactly.", flush=True)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if args.materialize:
        args.target_dir.mkdir(parents=True, exist_ok=True)
    calibration_records = []
    for current_id in selected_ids:
        source = candidates[current_id]
        target = args.target_dir / source.name
        if args.materialize:
            if not target.is_file() or sha256(target) != candidate_hashes[current_id]:
                shutil.copy2(source, target)
        calibration_records.append(
            {
                "split": "calibration",
                "image": relative_path(target, args.manifest),
                "image_sha256": candidate_hashes[current_id],
                "source": "coco_train2017",
                "image_id": current_id,
                "source_url": f"http://images.cocodataset.org/train2017/{source.name}",
            }
        )

    validation_document, validation_records = load_validation_records(args.validation_manifest)
    normalized_validation = []
    for record in validation_records:
        normalized = dict(record)
        normalized["image"] = relative_path(
            source_path(args.validation_manifest, record["image"]), args.manifest
        )
        normalized["label"] = relative_path(
            source_path(args.validation_manifest, record["label"]), args.manifest
        )
        normalized_validation.append(normalized)

    manifest = {
        "schema_version": 1,
        "dataset_id": "coco2017-yolov8n-calibration-v3-val5000-human-labels-v1",
        "annotation_format": "yolo_xywh_normalized",
        "calibration_count": len(calibration_records),
        "validation_count": len(normalized_validation),
        "selection": {
            "algorithm": EXPECTED_ALGORITHM,
            "algorithm_version": EXPECTED_ALGORITHM_VERSION,
            "candidate_count": len(candidates),
            "target_count": len(selected_ids),
            "old_baseline_membership_forced": False,
            "feature_names": list(FEATURE_NAMES),
            "selection_metadata": relative_path(args.selection_contract, args.manifest),
            "selection_metadata_sha256": sha256(args.selection_contract),
        },
        "records": calibration_records + normalized_validation,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = {
        "schema_version": 1,
        "status": "PREPARED",
        "selection_contract_verified": True,
        "dataset_id": manifest["dataset_id"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "selection_contract": str(args.selection_contract.resolve()),
        "selection_contract_sha256": sha256(args.selection_contract),
        "regenerated_selection": str(args.regenerated_selection.resolve()),
        "regenerated_selection_sha256": sha256(args.regenerated_selection),
        "annotations_sha256": annotation_hash,
        "validation_source_manifest_sha256": sha256(args.validation_manifest),
        "selection": manifest["selection"],
        "selected_ids": selected_ids,
        "coverage": summarize(selected_profiles, selected_features),
        "materialized": bool(args.materialize),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Calibration manifest: {args.manifest}")
    print(f"Coverage report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
