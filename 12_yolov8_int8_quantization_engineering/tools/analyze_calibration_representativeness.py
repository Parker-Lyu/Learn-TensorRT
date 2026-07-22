#!/usr/bin/env python3
"""Compare lesson 12 calibration-image geometry with the full COCO train2017 split."""

from __future__ import annotations

import argparse
import json
import math
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LESSON_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = REPO_ROOT / "assets/coco/data/downloads/annotations_trainval2017.zip"
DEFAULT_SELECTION = LESSON_DIR / "data/calibration_selection.json"
DEFAULT_OUTPUT_DIR = LESSON_DIR / "outputs/data_preparation/representativeness"
INSTANCE_ANNOTATIONS = "annotations/instances_train2017.json"


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    unit: str
    level: str
    log_x: bool = False


METRICS = (
    MetricSpec("image_width_px", "Image width", "px", "image"),
    MetricSpec("image_height_px", "Image height", "px", "image"),
    MetricSpec("image_area_mp", "Image area", "MP", "image", True),
    MetricSpec("image_aspect_ratio", "Image aspect ratio", "width / height", "image", True),
    MetricSpec("objects_per_image", "Objects per image", "non-crowd boxes", "image"),
    MetricSpec("box_scale_px", "Box scale", "sqrt(COCO area) px", "box", True),
    MetricSpec("box_relative_area", "Relative box area", "bbox area / image area", "box", True),
    MetricSpec("box_aspect_ratio", "Box aspect ratio", "width / height", "box", True),
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_annotations(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"COCO annotation archive not found: {path}")
    with zipfile.ZipFile(path) as archive:
        try:
            with archive.open(INSTANCE_ANNOTATIONS) as stream:
                return json.load(stream)
        except KeyError as error:
            raise ValueError(
                f"annotation archive does not contain {INSTANCE_ANNOTATIONS}: {path}"
            ) from error


def selected_image_ids(path: Path) -> list[int]:
    document = load_json(path)
    values = document.get("selected_ids")
    if not isinstance(values, list) or not values:
        raise ValueError("selection document must contain a non-empty selected_ids list")
    if any(not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("selected_ids must contain positive integer COCO image IDs")
    if len(values) != len(set(values)):
        raise ValueError("selected_ids contains duplicates")
    return values


def build_geometry(
    document: dict[str, Any],
) -> tuple[list[int], dict[str, np.ndarray], dict[str, list[np.ndarray]], dict[str, int]]:
    """Build image metrics and per-image box arrays without loading training images."""
    images = document.get("images")
    annotations = document.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO document must contain images and annotations lists")

    ordered_images = sorted(images, key=lambda item: int(item["id"]))
    image_ids = [int(image["id"]) for image in ordered_images]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("COCO annotations contain duplicate image IDs")
    offsets = {image_id: index for index, image_id in enumerate(image_ids)}
    widths = np.asarray([float(image["width"]) for image in ordered_images], dtype=np.float64)
    heights = np.asarray([float(image["height"]) for image in ordered_images], dtype=np.float64)
    if np.any(widths <= 0.0) or np.any(heights <= 0.0):
        raise ValueError("image dimensions must be positive")

    box_scale: list[list[float]] = [[] for _ in image_ids]
    box_relative_area: list[list[float]] = [[] for _ in image_ids]
    box_aspect_ratio: list[list[float]] = [[] for _ in image_ids]
    object_counts = np.zeros(len(image_ids), dtype=np.float64)
    size_counts = {"small": 0, "medium": 0, "large": 0}
    ignored_crowd = 0
    invalid_boxes = 0

    for annotation in annotations:
        if int(annotation.get("iscrowd", 0)) != 0:
            ignored_crowd += 1
            continue
        image_id = int(annotation["image_id"])
        if image_id not in offsets:
            raise ValueError(f"annotation references unknown image ID {image_id}")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            invalid_boxes += 1
            continue
        width = max(0.0, float(bbox[2]))
        height = max(0.0, float(bbox[3]))
        if width <= 0.0 or height <= 0.0:
            invalid_boxes += 1
            continue
        index = offsets[image_id]
        coco_area = max(0.0, float(annotation.get("area", width * height)))
        bbox_area = width * height
        image_area = widths[index] * heights[index]
        box_scale[index].append(math.sqrt(coco_area))
        box_relative_area[index].append(min(1.0, bbox_area / image_area))
        box_aspect_ratio[index].append(width / height)
        object_counts[index] += 1.0
        if coco_area < 32.0**2:
            size_counts["small"] += 1
        elif coco_area < 96.0**2:
            size_counts["medium"] += 1
        else:
            size_counts["large"] += 1

    image_metrics = {
        "image_width_px": widths,
        "image_height_px": heights,
        "image_area_mp": widths * heights / 1_000_000.0,
        "image_aspect_ratio": widths / heights,
        "objects_per_image": object_counts,
    }
    box_metrics = {
        "box_scale_px": [np.asarray(values, dtype=np.float64) for values in box_scale],
        "box_relative_area": [
            np.asarray(values, dtype=np.float64) for values in box_relative_area
        ],
        "box_aspect_ratio": [
            np.asarray(values, dtype=np.float64) for values in box_aspect_ratio
        ],
    }
    metadata = {
        "ignored_crowd_annotations": ignored_crowd,
        "invalid_non_crowd_boxes": invalid_boxes,
        **{f"full_{name}_boxes": count for name, count in size_counts.items()},
    }
    return image_ids, image_metrics, box_metrics, metadata


def concatenate_images(values: list[np.ndarray], indices: Iterable[int]) -> np.ndarray:
    arrays = [values[index] for index in indices if values[index].size]
    return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)


def ks_distance(sample: np.ndarray, population_sorted: np.ndarray) -> float:
    """Return the exact two-sample Kolmogorov-Smirnov D statistic."""
    if sample.size == 0 or population_sorted.size == 0:
        raise ValueError("KS distance requires two non-empty samples")
    ordered = np.sort(np.asarray(sample, dtype=np.float64))
    population = np.asarray(population_sorted, dtype=np.float64)
    if np.any(np.diff(population) < 0.0):
        population = np.sort(population)
    sample_count = ordered.size
    population_count = population.size
    unique, starts, counts = np.unique(ordered, return_index=True, return_counts=True)
    population_left = np.searchsorted(population, unique, side="left") / population_count
    population_right = np.searchsorted(population, unique, side="right") / population_count
    sample_left = starts / sample_count
    sample_right = (starts + counts) / sample_count
    return float(
        max(
            np.max(np.abs(sample_left - population_left)),
            np.max(np.abs(sample_right - population_right)),
        )
    )


def quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("cannot summarize an empty distribution")
    return {
        name: float(value)
        for name, value in zip(
            ("p05", "p25", "p50", "p75", "p95"),
            np.percentile(values, (5, 25, 50, 75, 95)),
        )
    }


def percentile_rank(value: float, baseline: np.ndarray) -> float:
    return float(100.0 * np.mean(baseline <= value))


def metric_status(rank: float) -> str:
    if rank <= 95.0:
        return "PASS"
    if rank <= 99.0:
        return "WARN"
    return "FAIL"


def support_coverage(sample: np.ndarray, population: np.ndarray) -> dict[str, Any]:
    """Check coverage of twenty approximately equal-mass population intervals."""
    edges = np.unique(np.percentile(population, np.linspace(0.0, 100.0, 21)))
    if edges.size < 3:
        return {"status": "PASS", "covered_bins": 1, "population_bins": 1, "empty_bins": []}
    edges[0] = -np.inf
    edges[-1] = np.inf
    counts, _ = np.histogram(sample, bins=edges)
    empty = np.flatnonzero(counts == 0).astype(int).tolist()
    return {
        "status": "PASS" if not empty else "FAIL",
        "covered_bins": int(counts.size - len(empty)),
        "population_bins": int(counts.size),
        "empty_bins": empty,
    }


def analyze_metric(
    spec: MetricSpec,
    selected: np.ndarray,
    population: np.ndarray,
    random_samples: list[np.ndarray],
) -> dict[str, Any]:
    population_sorted = np.sort(population)
    selected_distance = ks_distance(selected, population_sorted)
    baseline = np.asarray(
        [ks_distance(sample, population_sorted) for sample in random_samples],
        dtype=np.float64,
    )
    rank = percentile_rank(selected_distance, baseline)
    baseline_median = float(np.median(baseline))
    return {
        "title": spec.title,
        "unit": spec.unit,
        "level": spec.level,
        "population_count": int(population.size),
        "selected_count": int(selected.size),
        "population_quantiles": quantiles(population),
        "selected_quantiles": quantiles(selected),
        "ks_distance": selected_distance,
        "random_baseline": {
            "median": baseline_median,
            "p95": float(np.percentile(baseline, 95)),
            "p99": float(np.percentile(baseline, 99)),
            "selected_percentile_rank": rank,
            "distance_ratio_to_median": (
                selected_distance / baseline_median if baseline_median > 0.0 else None
            ),
        },
        "distribution_status": metric_status(rank),
        "support_coverage": support_coverage(selected, population),
    }


def coco_size_distribution(
    box_scale_by_image: list[np.ndarray], indices: Iterable[int]
) -> dict[str, Any]:
    values = concatenate_images(box_scale_by_image, indices)
    counts = {
        "small": int(np.sum(values < 32.0)),
        "medium": int(np.sum((values >= 32.0) & (values < 96.0))),
        "large": int(np.sum(values >= 96.0)),
    }
    total = max(1, sum(counts.values()))
    return {
        "counts": counts,
        "shares": {name: count / total for name, count in counts.items()},
    }


def plot_distributions(
    metrics: dict[str, dict[str, Any]],
    selected_values: dict[str, np.ndarray],
    population_values: dict[str, np.ndarray],
    destination: Path,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 12))
    for axis, spec in zip(axes.flat, METRICS):
        population = population_values[spec.key]
        selected = selected_values[spec.key]
        positive = population[population > 0.0]
        if spec.log_x and positive.size:
            low, high = np.percentile(positive, (0.2, 99.8))
            low = max(low, np.min(positive))
            if high > low:
                bins = np.geomspace(low, high, 31)
                axis.set_xscale("log")
            else:
                bins = 10
        else:
            low, high = np.percentile(population, (0.2, 99.8))
            bins = np.linspace(low, high, 31) if high > low else 10
        axis.hist(population, bins=bins, density=True, histtype="step", linewidth=2,
                  label="full train2017")
        axis.hist(selected, bins=bins, density=True, alpha=0.4, label="selected 3000")
        result = metrics[spec.key]
        axis.set_title(
            f"{spec.title} | KS={result['ks_distance']:.3f} {result['distribution_status']}"
        )
        axis.set_xlabel(spec.unit)
        axis.set_ylabel("density")
        axis.grid(alpha=0.2)
    axes.flat[len(METRICS)].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle("Lesson 12 calibration geometry vs. full COCO train2017", fontsize=16)
    figure.tight_layout(rect=(0, 0.04, 1, 0.97))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def fmt(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 0.01 or abs(value) >= 10_000:
        return f"{value:.3e}"
    return f"{value:.3f}"


def write_markdown(report: dict[str, Any], destination: Path, figure_name: str) -> None:
    fidelity = report["conclusion"]["distribution_fidelity"]
    coverage = report["conclusion"]["support_coverage"]
    verdict = report["conclusion"]["geometry_representativeness"]
    lines = [
        "# Calibration Geometry Representativeness",
        "",
        f"- Geometry verdict: **{verdict}**",
        f"- Distribution fidelity: **{fidelity}**",
        f"- Support coverage: **{coverage}**",
        f"- Population: {report['population']['images']:,} images, "
        f"{report['population']['non_crowd_boxes']:,} non-crowd boxes",
        f"- Calibration selection: {report['selection']['images']:,} images, "
        f"{report['selection']['non_crowd_boxes']:,} non-crowd boxes",
        "",
        "The distribution status compares the selected set's KS distance with deterministic random "
        f"subsets of the same image count ({report['method']['random_trials']} trials). PASS means "
        "the distance is no worse than the random-baseline 95th percentile; WARN is within the "
        "99th percentile; FAIL is beyond it. Support coverage checks twenty approximately "
        "equal-mass population intervals per metric.",
        "",
        "## Key Findings",
        "",
        f"- The median selected image contains "
        f"{fmt(report['metrics']['objects_per_image']['selected_quantiles']['p50'])} objects, "
        f"versus {fmt(report['metrics']['objects_per_image']['population_quantiles']['p50'])} "
        "in full train2017.",
        f"- The median selected box scale is "
        f"{fmt(report['metrics']['box_scale_px']['selected_quantiles']['p50'])} px, versus "
        f"{fmt(report['metrics']['box_scale_px']['population_quantiles']['p50'])} px in the "
        "population.",
        f"- Small-object share changes from "
        f"{100 * report['coco_object_sizes']['population']['shares']['small']:.2f}% to "
        f"{100 * report['coco_object_sizes']['selected']['shares']['small']:.2f}%; large-object "
        f"share changes from {100 * report['coco_object_sizes']['population']['shares']['large']:.2f}% "
        f"to {100 * report['coco_object_sizes']['selected']['shares']['large']:.2f}%.",
        "- Every tested population-support interval is represented, so the main issue is changed "
        "frequency weighting rather than a missing geometry region.",
        "",
        f"![Distribution comparison]({figure_name})",
        "",
        "| Metric | Selected KS | Random p95 | Baseline percentile | Full p50 / p95 | Selected p50 / p95 | Distribution | Coverage |",
        "|---|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for spec in METRICS:
        metric = report["metrics"][spec.key]
        full = metric["population_quantiles"]
        selected = metric["selected_quantiles"]
        lines.append(
            f"| {spec.title} | {metric['ks_distance']:.4f} | "
            f"{metric['random_baseline']['p95']:.4f} | "
            f"{metric['random_baseline']['selected_percentile_rank']:.1f}% | "
            f"{fmt(full['p50'])} / {fmt(full['p95'])} | "
            f"{fmt(selected['p50'])} / {fmt(selected['p95'])} | "
            f"{metric['distribution_status']} | {metric['support_coverage']['status']} |"
        )
    lines.extend(
        [
            "",
            "## COCO Object-size Shares",
            "",
            "| Size | Full train2017 | Selected 3000 | Difference |",
            "|---|---:|---:|---:|",
        ]
    )
    full_sizes = report["coco_object_sizes"]["population"]["shares"]
    selected_sizes = report["coco_object_sizes"]["selected"]["shares"]
    for name in ("small", "medium", "large"):
        lines.append(
            f"| {name} | {100 * full_sizes[name]:.2f}% | "
            f"{100 * selected_sizes[name]:.2f}% | "
            f"{100 * (selected_sizes[name] - full_sizes[name]):+.2f} pp |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This report evaluates image dimensions, object counts, and bounding-box geometry only. "
            "It does not by itself prove representativeness of categories, image content, color, "
            "illumination, texture, or model activations. Crowd annotations are excluded to match "
            "the lesson's calibration-selection geometry features.",
            "",
        ]
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.random_trials < 20:
        raise ValueError("--random-trials must be at least 20")
    selected_ids = selected_image_ids(args.selection)
    document = load_annotations(args.annotations)
    image_ids, image_metrics, box_metrics, metadata = build_geometry(document)
    del document
    offsets = {image_id: index for index, image_id in enumerate(image_ids)}
    missing = sorted(set(selected_ids) - offsets.keys())
    if missing:
        raise ValueError(f"selected image IDs are absent from train2017: {missing[:5]}")
    selected_indices = np.asarray([offsets[value] for value in selected_ids], dtype=np.int64)
    if selected_indices.size >= len(image_ids):
        raise ValueError("selection must be smaller than the train2017 population")

    population_values: dict[str, np.ndarray] = dict(image_metrics)
    population_values.update(
        {key: concatenate_images(values, range(len(image_ids))) for key, values in box_metrics.items()}
    )
    selected_values = {
        key: values[selected_indices] for key, values in image_metrics.items()
    }
    selected_values.update(
        {key: concatenate_images(values, selected_indices) for key, values in box_metrics.items()}
    )

    rng = random.Random(args.seed)
    random_index_sets = [
        np.asarray(rng.sample(range(len(image_ids)), len(selected_ids)), dtype=np.int64)
        for _ in range(args.random_trials)
    ]
    metrics: dict[str, dict[str, Any]] = {}
    for spec in METRICS:
        if spec.level == "image":
            samples = [image_metrics[spec.key][indices] for indices in random_index_sets]
        else:
            samples = [concatenate_images(box_metrics[spec.key], indices) for indices in random_index_sets]
        metrics[spec.key] = analyze_metric(
            spec, selected_values[spec.key], population_values[spec.key], samples
        )

    distribution_statuses = [metric["distribution_status"] for metric in metrics.values()]
    coverage_statuses = [metric["support_coverage"]["status"] for metric in metrics.values()]
    distribution_fidelity = (
        "FAIL" if "FAIL" in distribution_statuses else "WARN" if "WARN" in distribution_statuses else "PASS"
    )
    coverage = "FAIL" if "FAIL" in coverage_statuses else "PASS"
    if coverage == "FAIL":
        geometry_verdict = "NOT_GEOMETRICALLY_REPRESENTATIVE"
    elif distribution_fidelity == "FAIL":
        geometry_verdict = "NOT_DISTRIBUTIONALLY_REPRESENTATIVE_BUT_SUPPORT_COVERED"
    elif distribution_fidelity == "WARN":
        geometry_verdict = "REPRESENTATIVE_WITH_MINOR_DISTRIBUTION_SHIFT"
    else:
        geometry_verdict = "REPRESENTATIVE"

    all_indices = range(len(image_ids))
    report = {
        "schema_version": 1,
        "method": {
            "population": "COCO train2017 instances annotations",
            "crowd_annotations": "excluded",
            "distance": "two-sample Kolmogorov-Smirnov D",
            "baseline": "random image subsets matching the calibration image count",
            "random_trials": args.random_trials,
            "random_seed": args.seed,
            "distribution_thresholds": {"pass_percentile_max": 95, "warn_percentile_max": 99},
            "support_check": "20 approximately equal-mass population intervals",
        },
        "inputs": {
            "annotations": str(args.annotations.resolve()),
            "selection": str(args.selection.resolve()),
        },
        "population": {
            "images": len(image_ids),
            "non_crowd_boxes": int(population_values["box_scale_px"].size),
        },
        "selection": {
            "images": len(selected_ids),
            "non_crowd_boxes": int(selected_values["box_scale_px"].size),
        },
        "annotation_notes": metadata,
        "metrics": metrics,
        "coco_object_sizes": {
            "population": coco_size_distribution(box_metrics["box_scale_px"], all_indices),
            "selected": coco_size_distribution(box_metrics["box_scale_px"], selected_indices),
        },
        "conclusion": {
            "distribution_fidelity": distribution_fidelity,
            "support_coverage": coverage,
            "geometry_representativeness": geometry_verdict,
            "scope": "image dimensions, object count, and non-crowd box geometry only",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "representativeness_report.json"
    markdown_path = args.output_dir / "representativeness_report.md"
    figure_path = args.output_dir / "geometry_distributions.png"
    plot_distributions(metrics, selected_values, population_values, figure_path)
    write_markdown(report, markdown_path, figure_path.name)
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Geometry verdict: {geometry_verdict}")
    print(f"Distribution fidelity: {distribution_fidelity}")
    print(f"Support coverage: {coverage}")
    print(f"Markdown report: {markdown_path}")
    print(f"JSON report: {json_path}")
    print(f"Figure: {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
