#!/usr/bin/env python3
"""Build a versioned 3,000-image calibration split with broader scene coverage."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
LESSON12 = REPO_ROOT / "12_yolov8_int8_calibration"
sys.path.insert(0, str(LESSON12))

from dataset_manifest import load_manifest, resolve_path, sha256  # noqa: E402

DEFAULT_BASE_MANIFEST = REPO_ROOT / "assets/coco/data/dataset_manifest.json"
DEFAULT_ANNOTATIONS_ARCHIVE = (
    REPO_ROOT / "assets/coco/data/downloads/annotations_trainval2017.zip"
)
DEFAULT_IMAGE_ROOT = (
    REPO_ROOT / "assets/coco/data/calibration/images/train2017_coverage_v2_pool"
)
DEFAULT_OUTPUT_DIR = LESSON12 / "outputs/precision_recovery/02_calibration_coverage"
TRAIN_IMAGE_URL = "http://images.cocodataset.org/train2017/{file_name}"
USER_AGENT = "Learn-TensorRT-precision-recovery/1.0"
SIZE_NAMES = ("small", "medium", "large")


def load_train_annotations(archive_path: Path) -> dict[str, Any]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"COCO annotation archive not found: {archive_path}")
    member = "annotations/instances_train2017.json"
    with zipfile.ZipFile(archive_path) as archive:
        try:
            with archive.open(member) as stream:
                return json.load(stream)
        except KeyError as error:
            raise RuntimeError(f"{member} is missing from {archive_path}") from error


def size_name(area: float) -> str:
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def build_profiles(document: dict[str, Any]) -> dict[int, dict[str, Any]]:
    images = {
        int(image["id"]): {
            "id": int(image["id"]),
            "file_name": str(image["file_name"]),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "classes": set(),
            "class_counts": Counter(),
            "size_counts": Counter(),
            "box_areas": [],
        }
        for image in document.get("images", [])
    }
    category_ids = sorted(int(category["id"]) for category in document.get("categories", []))
    class_index = {category_id: index for index, category_id in enumerate(category_ids)}
    if len(class_index) != 80:
        raise ValueError(f"expected 80 COCO categories, found {len(class_index)}")

    for annotation in document.get("annotations", []):
        if annotation.get("iscrowd", 0):
            continue
        profile = images.get(int(annotation["image_id"]))
        if profile is None:
            raise ValueError(f"annotation references unknown image {annotation['image_id']}")
        class_id = class_index[int(annotation["category_id"])]
        area = max(0.0, float(annotation.get("area", 0.0)))
        profile["classes"].add(class_id)
        profile["class_counts"][class_id] += 1
        profile["size_counts"][size_name(area)] += 1
        image_area = max(1.0, float(profile["width"] * profile["height"]))
        profile["box_areas"].append(area / image_area)

    for profile in images.values():
        profile["classes"] = tuple(sorted(profile["classes"]))
        profile["object_count"] = int(sum(profile["class_counts"].values()))
    return images


def image_id_from_path(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as error:
        raise ValueError(f"COCO image filename must contain its numeric ID: {path}") from error


def annotation_pool(
    profiles: dict[int, dict[str, Any]], excluded_ids: set[int], pool_size: int, seed: int
) -> list[int]:
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    eligible = [profile for image_id, profile in profiles.items() if image_id not in excluded_ids]
    if pool_size > len(eligible):
        raise ValueError(f"pool_size {pool_size} exceeds {len(eligible)} eligible images")

    rng = random.Random(seed)
    jitter = {profile["id"]: rng.random() for profile in eligible}
    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for profile in eligible:
        for class_id in profile["classes"]:
            by_class[class_id].append(profile)

    def priority(profile: dict[str, Any], class_id: int) -> tuple[Any, ...]:
        sizes = profile["size_counts"]
        size_diversity = sum(int(sizes[name] > 0) for name in SIZE_NAMES)
        return (
            -size_diversity,
            -int(profile["class_counts"][class_id]),
            -len(profile["classes"]),
            -min(profile["object_count"], 30),
            jitter[profile["id"]],
            profile["id"],
        )

    for class_id, values in by_class.items():
        values.sort(key=lambda profile, current=class_id: priority(profile, current))

    selected: list[int] = []
    selected_set: set[int] = set()
    offsets = Counter()
    classes = sorted(by_class)
    while len(selected) < pool_size and classes:
        made_progress = False
        for class_id in classes:
            values = by_class[class_id]
            while offsets[class_id] < len(values):
                image_id = values[offsets[class_id]]["id"]
                offsets[class_id] += 1
                if image_id not in selected_set:
                    selected.append(image_id)
                    selected_set.add(image_id)
                    made_progress = True
                    break
            if len(selected) == pool_size:
                break
        if not made_progress:
            break

    if len(selected) < pool_size:
        remaining = sorted(
            (profile for profile in eligible if profile["id"] not in selected_set),
            key=lambda profile: (
                -sum(int(profile["size_counts"][name] > 0) for name in SIZE_NAMES),
                -len(profile["classes"]),
                -min(profile["object_count"], 30),
                jitter[profile["id"]],
                profile["id"],
            ),
        )
        selected.extend(profile["id"] for profile in remaining[: pool_size - len(selected)])
    return selected


def download_image(file_name: str, destination: Path, retries: int = 3) -> Path:
    if destination.is_file():
        image = cv2.imread(str(destination), cv2.IMREAD_COLOR)
        if image is not None:
            return destination
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        TRAIN_IMAGE_URL.format(file_name=file_name), headers={"User-Agent": USER_AGENT}
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            partial.replace(destination)
            if cv2.imread(str(destination), cv2.IMREAD_COLOR) is None:
                raise RuntimeError(f"downloaded file is not a readable image: {destination}")
            return destination
        except (OSError, urllib.error.URLError, RuntimeError) as error:
            partial.unlink(missing_ok=True)
            destination.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"failed to download {file_name}: {error}") from error
            time.sleep(attempt)
    raise AssertionError("unreachable")


def download_pool(
    image_ids: list[int], profiles: dict[int, dict[str, Any]], root: Path, workers: int
) -> dict[int, Path]:
    if workers <= 0:
        raise ValueError("workers must be positive")

    def fetch(image_id: int) -> tuple[int, Path]:
        file_name = profiles[image_id]["file_name"]
        return image_id, download_image(file_name, root / file_name)

    paths: dict[int, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, image_id) for image_id in image_ids]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            image_id, path = future.result()
            paths[image_id] = path
            if completed % 100 == 0 or completed == len(futures):
                print(f"Candidate images ready: {completed}/{len(futures)}", flush=True)
    return paths


def numeric_features(profile: dict[str, Any], path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image for coverage features: {path}")
    height, width = image.shape[:2]
    scale = min(256.0 / max(width, 1), 256.0 / max(height, 1), 1.0)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny((gray * 255.0).astype(np.uint8), 80, 160)
    count = max(1, profile["object_count"])
    sizes = profile["size_counts"]
    box_areas = profile["box_areas"]
    return np.asarray([
        math.log1p(profile["object_count"]),
        sizes["small"] / count,
        sizes["medium"] / count,
        sizes["large"] / count,
        math.log(max(profile["width"], 1) / max(profile["height"], 1)),
        math.log1p(10_000.0 * (float(np.mean(box_areas)) if box_areas else 0.0)),
        float(np.mean(gray)),
        float(np.std(gray)),
        float(np.mean(gray < 0.15)),
        float(np.mean(gray > 0.85)),
        float(np.mean(hsv[..., 1])) / 255.0,
        float(np.mean(edges > 0)),
    ], dtype=np.float64)


def robust_scale(base: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined = np.vstack((base, candidates))
    low = np.percentile(combined, 5, axis=0)
    high = np.percentile(combined, 95, axis=0)
    span = np.maximum(high - low, 1e-9)
    return (
        np.clip((base - low) / span, -0.5, 1.5),
        np.clip((candidates - low) / span, -0.5, 1.5),
    )


def farthest_coverage_selection(
    base_features: np.ndarray, candidate_features: np.ndarray, count: int
) -> list[int]:
    if base_features.ndim != 2 or candidate_features.ndim != 2:
        raise ValueError("feature arrays must be two-dimensional")
    if base_features.shape[1] != candidate_features.shape[1]:
        raise ValueError("base and candidate feature dimensions differ")
    if count < 0 or count > len(candidate_features):
        raise ValueError("selection count is outside the candidate range")
    scaled_base, scaled_candidates = robust_scale(base_features, candidate_features)
    nearest = np.full(len(scaled_candidates), np.inf, dtype=np.float64)
    for start in range(0, len(scaled_base), 128):
        block = scaled_base[start:start + 128]
        distances = np.sum(
            (scaled_candidates[:, None, :] - block[None, :, :]) ** 2, axis=2
        )
        nearest = np.minimum(nearest, np.min(distances, axis=1))

    selected = []
    available = np.ones(len(scaled_candidates), dtype=bool)
    for _ in range(count):
        scores = np.where(available, nearest, -1.0)
        index = int(np.argmax(scores))
        selected.append(index)
        available[index] = False
        distance = np.sum((scaled_candidates - scaled_candidates[index]) ** 2, axis=1)
        nearest = np.minimum(nearest, distance)
    return selected


def summarize(profiles: list[dict[str, Any]], features: np.ndarray) -> dict[str, Any]:
    class_images = Counter(class_id for profile in profiles for class_id in profile["classes"])
    size_objects = Counter()
    for profile in profiles:
        size_objects.update(profile["size_counts"])
    return {
        "images": len(profiles),
        "class_image_occurrences": {str(index): class_images[index] for index in range(80)},
        "class_images_min": min(class_images.values()) if class_images else 0,
        "class_images_max": max(class_images.values()) if class_images else 0,
        "object_size_counts": {name: size_objects[name] for name in SIZE_NAMES},
        "object_count": {
            "mean": float(np.mean(features[:, 0])),
            "note": "mean log1p(object_count)",
        },
        "luma_mean": {
            "p05": float(np.percentile(features[:, 6], 5)),
            "p50": float(np.percentile(features[:, 6], 50)),
            "p95": float(np.percentile(features[:, 6], 95)),
        },
        "luma_contrast": {
            "p05": float(np.percentile(features[:, 7], 5)),
            "p50": float(np.percentile(features[:, 7], 50)),
            "p95": float(np.percentile(features[:, 7], 95)),
        },
        "image_aspect_log": {
            "min": float(np.min(features[:, 4])),
            "max": float(np.max(features[:, 4])),
        },
    }


def write_manifest(
    output: Path,
    dataset_id: str,
    calibration_paths: list[Path],
    base_manifest_path: Path,
    base_manifest: dict[str, Any],
) -> dict[str, Any]:
    records = [
        {
            "split": "calibration",
            "image": str(path.resolve()),
            "image_sha256": sha256(path),
        }
        for path in calibration_paths
    ]
    for record in base_manifest["records"]:
        if record["split"] != "validation":
            continue
        records.append({
            **record,
            "image": str(resolve_path(base_manifest_path, record["image"]).resolve()),
            "label": str(resolve_path(base_manifest_path, record["label"]).resolve()),
        })
    document = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "annotation_format": "yolo_xywh_normalized",
        "calibration_count": len(calibration_paths),
        "validation_count": base_manifest["validation_count"],
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    load_manifest(output)
    return document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--annotations-archive", type=Path, default=DEFAULT_ANNOTATIONS_ARCHIVE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-count", type=int, default=3000)
    parser.add_argument("--candidate-pool-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.target_count <= 1000:
        raise ValueError("target-count must be greater than the 1,000-image baseline")
    if args.candidate_pool_size < args.target_count - 1000:
        raise ValueError("candidate pool is smaller than the requested number of additions")

    print("Loading pinned COCO train2017 annotations...", flush=True)
    annotations = load_train_annotations(args.annotations_archive)
    profiles = build_profiles(annotations)
    del annotations

    base_manifest = load_manifest(args.base_manifest)
    base_records = [
        record for record in base_manifest["records"] if record["split"] == "calibration"
    ]
    if len(base_records) != 1000:
        raise ValueError(f"expected a 1,000-image baseline, found {len(base_records)}")
    base_paths = [resolve_path(args.base_manifest, record["image"]) for record in base_records]
    base_ids = [image_id_from_path(path) for path in base_paths]
    missing_profiles = sorted(set(base_ids) - profiles.keys())
    if missing_profiles:
        raise ValueError(f"baseline IDs are absent from train2017 annotations: {missing_profiles[:5]}")

    pool_ids = annotation_pool(profiles, set(base_ids), args.candidate_pool_size, args.seed)
    pool_paths_by_id = download_pool(pool_ids, profiles, args.image_root, args.workers)

    print("Computing annotation and image coverage features...", flush=True)
    base_features = np.vstack([
        numeric_features(profiles[image_id], path) for image_id, path in zip(base_ids, base_paths)
    ])
    pool_features = np.vstack([
        numeric_features(profiles[image_id], pool_paths_by_id[image_id]) for image_id in pool_ids
    ])
    selected_offsets = farthest_coverage_selection(
        base_features, pool_features, args.target_count - len(base_ids)
    )
    added_ids = [pool_ids[index] for index in selected_offsets]
    final_ids = base_ids + added_ids
    final_paths = base_paths + [pool_paths_by_id[image_id] for image_id in added_ids]
    final_features = np.vstack((base_features, pool_features[selected_offsets]))

    dataset_id = (
        f"coco2017-train{args.target_count}-coverage-v2-seed{args.seed}-"
        "calibration-val5000-human-labels-v1"
    )
    manifest_path = args.output_dir / "dataset_manifest.json"
    write_manifest(manifest_path, dataset_id, final_paths, args.base_manifest, base_manifest)

    report = {
        "schema_version": 1,
        "step": "02_calibration_coverage",
        "status": "PREPARED",
        "selection": {
            "dataset_id": dataset_id,
            "seed": args.seed,
            "baseline_images_retained": len(base_ids),
            "annotation_stratified_candidate_pool": len(pool_ids),
            "coverage_selected_additions": len(added_ids),
            "final_calibration_images": len(final_ids),
            "feature_names": [
                "log1p_object_count", "small_ratio", "medium_ratio", "large_ratio",
                "log_image_aspect", "log_mean_relative_box_area", "luma_mean",
                "luma_std", "dark_pixel_fraction", "bright_pixel_fraction",
                "mean_saturation", "edge_fraction",
            ],
            "method": (
                "per-class round-robin annotation pool followed by deterministic "
                "farthest-point sampling from the retained baseline in robust-scaled feature space"
            ),
        },
        "artifacts": {
            "base_manifest": str(args.base_manifest.resolve()),
            "base_manifest_sha256": sha256(args.base_manifest),
            "candidate_manifest": str(manifest_path.resolve()),
            "candidate_manifest_sha256": sha256(manifest_path),
            "annotations_archive": str(args.annotations_archive.resolve()),
            "annotations_archive_sha256": sha256(args.annotations_archive),
        },
        "coverage": {
            "baseline": summarize([profiles[image_id] for image_id in base_ids], base_features),
            "candidate": summarize([profiles[image_id] for image_id in final_ids], final_features),
        },
        "added_image_ids": added_ids,
    }
    report_path = args.output_dir / "coverage_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Candidate manifest: {manifest_path}")
    print(f"Coverage evidence: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
