#!/usr/bin/env python3
"""Select and materialize the canonical 80/20 COCO train2017 calibration set."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
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

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LESSON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LESSON_DIR))

from calibration_preprocessing import input_luma_mean  # noqa: E402


DEFAULT_ANNOTATIONS = REPO_ROOT / "assets/coco/data/downloads/annotations_trainval2017.zip"
DEFAULT_VALIDATION_MANIFEST = REPO_ROOT / "assets/coco/data/dataset_manifest.json"
DEFAULT_CONFIG = LESSON_DIR / "configs/calibration_selection.json"
DEFAULT_CONTRACT = LESSON_DIR / "data/calibration_selection.json"
DEFAULT_MANIFEST = LESSON_DIR / "data/dataset_manifest.json"
DEFAULT_IMAGE_CACHE = REPO_ROOT / "assets/coco/data/downloads/train2017_images"
DEFAULT_TARGET_DIR = (
    REPO_ROOT / "assets/coco/data/calibration/images/train2017_mixed_v4_n3000"
)
DEFAULT_REPORT = LESSON_DIR / "outputs/data_preparation/selection_report.json"
TRAIN_IMAGE_URL = "http://images.cocodataset.org/train2017/{file_name}"
USER_AGENT = "Learn-TensorRT-lesson12-calibration-selector/4.0"
ANNOTATION_MEMBER = "annotations/instances_train2017.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_config(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema_version") != 1:
        raise ValueError("selection config must use schema version 1")
    if document.get("algorithm_version") != 4:
        raise ValueError("selection config must use algorithm version 4")
    counts = document.get("counts", {})
    groups = document.get("tail_groups")
    if not isinstance(groups, list) or len(groups) != 6 or len(set(groups)) != 6:
        raise ValueError("selection config must declare six unique tail groups")
    if counts.get("natural_core") + counts.get("tail_total") != counts.get("total"):
        raise ValueError("natural and tail counts must sum to the total")
    if counts.get("per_tail_group") * len(groups) != counts.get("tail_total"):
        raise ValueError("tail group quotas must sum to tail_total")
    shape = document.get("input_shape")
    if shape != [1, 3, 640, 640]:
        raise ValueError("selection preprocessing must use 1x3x640x640")
    return document


def load_annotations(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"COCO annotation archive not found: {path}")
    with zipfile.ZipFile(path) as archive:
        try:
            with archive.open(ANNOTATION_MEMBER) as stream:
                return json.load(stream)
        except KeyError as error:
            raise ValueError(f"archive does not contain {ANNOTATION_MEMBER}") from error


def build_profiles(document: dict[str, Any]) -> dict[int, dict[str, Any]]:
    category_ids = sorted(int(category["id"]) for category in document["categories"])
    if len(category_ids) != 80:
        raise ValueError(f"expected 80 COCO categories, found {len(category_ids)}")
    class_index = {category_id: index for index, category_id in enumerate(category_ids)}
    profiles = {
        int(image["id"]): {
            "image_id": int(image["id"]),
            "file_name": str(image["file_name"]),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "classes": set(),
            "object_count": 0,
            "relative_areas": [],
            "box_aspect_scores": [],
        }
        for image in document["images"]
    }
    for annotation in document["annotations"]:
        if int(annotation.get("iscrowd", 0)) != 0:
            continue
        profile = profiles[int(annotation["image_id"])]
        _, _, box_width, box_height = (float(value) for value in annotation["bbox"])
        if box_width <= 0.0 or box_height <= 0.0:
            continue
        image_area = max(1.0, float(profile["width"] * profile["height"]))
        profile["classes"].add(class_index[int(annotation["category_id"])])
        profile["object_count"] += 1
        profile["relative_areas"].append(min(1.0, box_width * box_height / image_area))
        profile["box_aspect_scores"].append(abs(math_log_ratio(box_width, box_height)))
    for profile in profiles.values():
        relative_areas = profile.pop("relative_areas")
        box_scores = profile.pop("box_aspect_scores")
        profile["classes"] = tuple(sorted(profile["classes"]))
        profile["min_relative_box_area"] = min(relative_areas, default=1.0)
        profile["max_relative_box_area"] = max(relative_areas, default=0.0)
        profile["aspect_score"] = max(
            abs(math_log_ratio(profile["width"], profile["height"])),
            max(box_scores, default=0.0),
        )
    return profiles


def math_log_ratio(left: float, right: float) -> float:
    return float(np.log(max(left, 1.0e-12) / max(right, 1.0e-12)))


def natural_core(
    profiles: dict[int, dict[str, Any]], count: int, seed: int
) -> tuple[list[int], list[dict[str, int]]]:
    ordered_ids = sorted(profiles)
    selected = random.Random(seed).sample(ordered_ids, count)
    selected_set = set(selected)
    class_counts = Counter(
        class_id for image_id in selected for class_id in profiles[image_id]["classes"]
    )
    swaps: list[dict[str, int]] = []
    missing = [class_id for class_id in range(80) if class_counts[class_id] == 0]
    for class_id in missing:
        if class_counts[class_id] > 0:
            continue
        replacement = min(
            image_id
            for image_id, profile in profiles.items()
            if image_id not in selected_set and class_id in profile["classes"]
        )
        removable = [
            image_id
            for image_id in selected
            if all(class_counts[value] > 1 for value in profiles[image_id]["classes"])
        ]
        if not removable:
            raise ValueError(f"cannot repair natural-core coverage for class {class_id}")
        removed = min(
            removable,
            key=lambda image_id: (
                len(profiles[image_id]["classes"]),
                profiles[image_id]["object_count"],
                image_id,
            ),
        )
        selected[selected.index(removed)] = replacement
        selected_set.remove(removed)
        selected_set.add(replacement)
        class_counts.subtract(profiles[removed]["classes"])
        class_counts.update(profiles[replacement]["classes"])
        swaps.append({"removed": removed, "added": replacement, "class_id": class_id})
    return selected, swaps


def percentile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot compute a percentile from an empty population")
    return float(np.quantile(array, quantile))


def choose_geometry_tail(
    profiles: dict[int, dict[str, Any]],
    excluded: set[int],
    group: str,
    count: int,
    quantile: float,
    seed: int,
) -> tuple[list[int], float]:
    available = [profile for image_id, profile in profiles.items() if image_id not in excluded]
    with_objects = [profile for profile in available if profile["object_count"] > 0]
    if group == "small_object":
        threshold = percentile(
            (profile["min_relative_box_area"] for profile in with_objects), quantile
        )
        eligible = [
            profile["image_id"]
            for profile in with_objects
            if profile["min_relative_box_area"] <= threshold
        ]
    elif group == "large_object":
        threshold = percentile(
            (profile["max_relative_box_area"] for profile in with_objects), quantile
        )
        eligible = [
            profile["image_id"]
            for profile in with_objects
            if profile["max_relative_box_area"] >= threshold
        ]
    elif group == "crowded":
        threshold = percentile((profile["object_count"] for profile in available), quantile)
        eligible = [
            profile["image_id"]
            for profile in available
            if profile["object_count"] >= threshold
        ]
    elif group == "extreme_aspect":
        threshold = percentile((profile["aspect_score"] for profile in available), quantile)
        eligible = [
            profile["image_id"]
            for profile in available
            if profile["aspect_score"] >= threshold
        ]
    else:
        raise ValueError(f"unsupported geometry tail group: {group}")
    if len(eligible) < count:
        raise ValueError(f"tail group {group} has only {len(eligible)} eligible images")
    return random.Random(seed).sample(sorted(eligible), count), threshold


def download_file(url: str, destination: Path, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            partial.replace(destination)
            return
        except (OSError, urllib.error.URLError) as error:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {error}") from error
            time.sleep(attempt)


def ensure_image(
    profile: dict[str, Any],
    cache_dir: Path,
    expected_hash: str | None = None,
    downloader: Callable[[str, Path], None] = download_file,
) -> tuple[Path, str]:
    destination = cache_dir / profile["file_name"]
    if destination.is_file():
        actual_hash = sha256(destination)
        if expected_hash is None or actual_hash == expected_hash:
            return destination, actual_hash
        destination.unlink()
    file_name = urllib.parse.quote(profile["file_name"])
    downloader(TRAIN_IMAGE_URL.format(file_name=file_name), destination)
    actual_hash = sha256(destination)
    if expected_hash is not None and actual_hash != expected_hash:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded image hash mismatch: {profile['file_name']}")
    return destination, actual_hash


def ensure_images(
    image_ids: Iterable[int],
    profiles: dict[int, dict[str, Any]],
    cache_dir: Path,
    workers: int,
    expected_hashes: dict[int, str] | None = None,
) -> dict[int, tuple[Path, str]]:
    ordered = sorted(set(image_ids))
    if workers <= 0:
        raise ValueError("--workers must be positive")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                ensure_image,
                profiles[image_id],
                cache_dir,
                (expected_hashes or {}).get(image_id),
            ): image_id
            for image_id in ordered
        }
        results = {}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results[futures[future]] = future.result()
            if completed % 250 == 0 or completed == len(futures):
                print(f"  train2017 images: {completed}/{len(futures)}", flush=True)
    return results


def brightness_tail(
    screen_ids: list[int],
    paths: dict[int, tuple[Path, str]],
    input_shape: tuple[int, int, int, int],
    count: int,
    dark_quantile: float,
    bright_quantile: float,
    seed: int,
) -> tuple[list[int], list[int], dict[int, float], dict[str, float]]:
    luma = {image_id: input_luma_mean(paths[image_id][0], input_shape) for image_id in screen_ids}
    dark_threshold = percentile(luma.values(), dark_quantile)
    bright_threshold = percentile(luma.values(), bright_quantile)
    dark_candidates = sorted(image_id for image_id, value in luma.items() if value <= dark_threshold)
    bright_candidates = sorted(image_id for image_id, value in luma.items() if value >= bright_threshold)
    dark = random.Random(seed).sample(dark_candidates, count)
    bright = random.Random(seed + 1).sample(
        sorted(set(bright_candidates) - set(dark)), count
    )
    return dark, bright, luma, {"dark": dark_threshold, "bright": bright_threshold}


def selection_from_source(
    profiles: dict[int, dict[str, Any]],
    config: dict[str, Any],
    cache_dir: Path,
    workers: int,
) -> tuple[dict[str, list[int]], dict[str, Any], dict[int, tuple[Path, str]], dict[int, float]]:
    counts = config["counts"]
    quantiles = config["tail_quantiles"]
    seed = int(config["random_seed"])
    core, swaps = natural_core(profiles, int(counts["natural_core"]), seed)
    excluded = set(core)
    groups: dict[str, list[int]] = {"natural_core": core}
    thresholds: dict[str, float] = {}
    for offset, group in enumerate(
        ("small_object", "large_object", "crowded", "extreme_aspect"), start=1
    ):
        chosen, threshold = choose_geometry_tail(
            profiles,
            excluded,
            group,
            int(counts["per_tail_group"]),
            float(quantiles[group]),
            seed + offset,
        )
        groups[group] = chosen
        thresholds[group] = threshold
        excluded.update(chosen)

    remaining = sorted(set(profiles) - excluded)
    screen_ids = random.Random(seed + 5).sample(remaining, int(counts["brightness_screen"]))
    screen_paths = ensure_images(screen_ids, profiles, cache_dir, workers)
    dark, bright, luma, brightness_thresholds = brightness_tail(
        screen_ids,
        screen_paths,
        tuple(config["input_shape"]),
        int(counts["per_tail_group"]),
        float(quantiles["dark"]),
        float(quantiles["bright"]),
        seed + 6,
    )
    groups["dark"] = dark
    groups["bright"] = bright
    thresholds.update(brightness_thresholds)
    final_ids = [image_id for group in ["natural_core", *config["tail_groups"]] for image_id in groups[group]]
    if len(final_ids) != int(counts["total"]) or len(final_ids) != len(set(final_ids)):
        raise RuntimeError("selection did not produce the configured number of unique images")
    final_paths = ensure_images(final_ids, profiles, cache_dir, workers)
    all_paths = {**screen_paths, **final_paths}
    metadata = {
        "category_coverage_swaps": swaps,
        "thresholds": thresholds,
        "brightness_screen_ids": screen_ids,
    }
    return groups, metadata, all_paths, luma


def contract_document(
    config: dict[str, Any],
    annotation_hash: str,
    config_hash: str,
    profiles: dict[int, dict[str, Any]],
    groups: dict[str, list[int]],
    metadata: dict[str, Any],
    paths: dict[int, tuple[Path, str]],
    luma: dict[int, float],
) -> dict[str, Any]:
    role_by_id = {
        image_id: role for role, image_ids in groups.items() for image_id in image_ids
    }
    selected_ids = [
        image_id
        for role in ["natural_core", *config["tail_groups"]]
        for image_id in groups[role]
    ]
    return {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "algorithm": config["algorithm"],
        "algorithm_version": config["algorithm_version"],
        "annotations_sha256": annotation_hash,
        "config_sha256": config_hash,
        "category_coverage_swaps": metadata["category_coverage_swaps"],
        "thresholds": metadata["thresholds"],
        "brightness_screen": [
            {
                "image_id": image_id,
                "file_name": profiles[image_id]["file_name"],
                "image_sha256": paths[image_id][1],
                "input_luma_mean": luma[image_id],
            }
            for image_id in metadata["brightness_screen_ids"]
        ],
        "selected_images": [
            {
                "image_id": image_id,
                "file_name": profiles[image_id]["file_name"],
                "image_sha256": paths[image_id][1],
                "role": role_by_id[image_id],
                **(
                    {"input_luma_mean": luma[image_id]}
                    if image_id in luma
                    else {}
                ),
            }
            for image_id in selected_ids
        ],
    }


def verify_contract(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    if expected != actual:
        expected_ids = [record["image_id"] for record in expected.get("selected_images", [])]
        actual_ids = [record["image_id"] for record in actual.get("selected_images", [])]
        for index, (left, right) in enumerate(zip(expected_ids, actual_ids)):
            if left != right:
                raise RuntimeError(
                    f"selection contract mismatch at position {index}: expected {left}, got {right}"
                )
        raise RuntimeError("selection metadata or image content differs from the committed contract")


def materialize(
    contract: dict[str, Any], cache_dir: Path, target_dir: Path
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    expected_names = {record["file_name"] for record in contract["selected_images"]}
    for stale in target_dir.glob("*.jpg"):
        if stale.name not in expected_names:
            stale.unlink()
    for record in contract["selected_images"]:
        source = cache_dir / record["file_name"]
        target = target_dir / record["file_name"]
        if target.is_file() and sha256(target) == record["image_sha256"]:
            continue
        target.unlink(missing_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def relative_path(target: Path, document_path: Path) -> str:
    return os.path.relpath(target.resolve(), document_path.parent.resolve())


def build_manifest(
    contract: dict[str, Any],
    config: dict[str, Any],
    target_dir: Path,
    validation_manifest: Path,
    output: Path,
    contract_path: Path,
) -> dict[str, Any]:
    validation = load_json(validation_manifest)
    validation_records = [
        record for record in validation["records"] if record["split"] == "validation"
    ]
    calibration_hashes = {record["image_sha256"] for record in contract["selected_images"]}
    validation_hashes = {record["image_sha256"] for record in validation_records}
    overlap = calibration_hashes & validation_hashes
    if overlap:
        raise ValueError(
            f"calibration and validation overlap by {len(overlap)} image hash(es)"
        )
    calibration_records = [
        {
            "split": "calibration",
            "image": relative_path(target_dir / record["file_name"], output),
            "image_sha256": record["image_sha256"],
            "source": "coco_train2017",
            "image_id": record["image_id"],
            "selection_role": record["role"],
        }
        for record in contract["selected_images"]
    ]
    normalized_validation = []
    for record in validation_records:
        normalized = dict(record)
        for key in ("image", "label"):
            source = Path(record[key])
            if not source.is_absolute():
                source = validation_manifest.parent / source
            normalized[key] = relative_path(source, output)
        normalized_validation.append(normalized)
    return {
        "schema_version": 1,
        "dataset_id": config["dataset_id"],
        "annotation_format": "yolo_xywh_normalized",
        "calibration_count": len(calibration_records),
        "validation_count": len(normalized_validation),
        "selection": {
            "algorithm": config["algorithm"],
            "algorithm_version": config["algorithm_version"],
            "natural_core_count": config["counts"]["natural_core"],
            "tail_count": config["counts"]["tail_total"],
            "selection_metadata": relative_path(contract_path, output),
            "selection_metadata_sha256": sha256(contract_path),
        },
        "records": calibration_records + normalized_validation,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-cache", type=Path, default=DEFAULT_IMAGE_CACHE)
    parser.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--create-contract",
        action="store_true",
        help="Create the release contract; normal course runs verify it without modification.",
    )
    parser.add_argument("--materialize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = validate_config(load_json(args.config))
    annotation_hash = sha256(args.annotations)
    profiles = build_profiles(load_annotations(args.annotations))
    groups, metadata, paths, luma = selection_from_source(
        profiles, config, args.image_cache, args.workers
    )
    regenerated = contract_document(
        config,
        annotation_hash,
        sha256(args.config),
        profiles,
        groups,
        metadata,
        paths,
        luma,
    )
    if args.create_contract:
        if args.contract.exists():
            raise FileExistsError(f"refusing to overwrite selection contract: {args.contract}")
        args.contract.parent.mkdir(parents=True, exist_ok=True)
        args.contract.write_text(json.dumps(regenerated, indent=2) + "\n", encoding="utf-8")
    else:
        verify_contract(load_json(args.contract), regenerated)
    contract = load_json(args.contract)
    if args.materialize:
        materialize(contract, args.image_cache, args.target_dir)
    manifest = build_manifest(
        contract,
        config,
        args.target_dir,
        args.validation_manifest,
        args.manifest,
        args.contract,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "dataset_id": config["dataset_id"],
        "annotations_sha256": annotation_hash,
        "selection_contract_sha256": sha256(args.contract),
        "manifest_sha256": sha256(args.manifest),
        "counts": {role: len(image_ids) for role, image_ids in groups.items()},
        "thresholds": metadata["thresholds"],
        "classes_covered": len(
            {
                class_id
                for image_ids in groups.values()
                for image_id in image_ids
                for class_id in profiles[image_id]["classes"]
            }
        ),
        "calibration_validation_overlap": 0,
        "materialized": args.materialize,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Selection contract verified: {args.contract}")
    print(f"Calibration manifest: {args.manifest}")
    print(f"Selection report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
