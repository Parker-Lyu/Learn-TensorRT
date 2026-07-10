"""Pure dataset-evaluation helpers shared by lesson 12 runners and tests."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def load_yolo_labels(path: Path, image_width: int, image_height: int) -> list[dict]:
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    labels = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected class cx cy width height")
        class_id = int(parts[0])
        cx, cy, width, height = (float(value) for value in parts[1:])
        if class_id < 0 or not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height)):
            raise ValueError(f"{path}:{line_number}: invalid normalized YOLO annotation")
        labels.append({
            "class_id": class_id,
            "box_xyxy": [
                (cx - width * 0.5) * image_width,
                (cy - height * 0.5) * image_height,
                (cx + width * 0.5) * image_width,
                (cy + height * 0.5) * image_height,
            ],
        })
    return labels


def box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    a = list(left)
    b = list(right)
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1.0e-12)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated = [np.max(precision[recall >= point], initial=0.0) for point in recall_points]
    return float(np.mean(interpolated))


def match_class(predictions: list[dict],
                ground_truth: dict[str, list[dict]],
                class_id: int,
                iou_threshold: float) -> tuple[np.ndarray, np.ndarray, int]:
    class_predictions = sorted(
        (item for item in predictions if item["class_id"] == class_id),
        key=lambda item: item["confidence"],
        reverse=True,
    )
    class_truth = {
        image: [box for box in boxes if box["class_id"] == class_id]
        for image, boxes in ground_truth.items()
    }
    truth_count = sum(len(boxes) for boxes in class_truth.values())
    matched = {image: set() for image in class_truth}
    true_positive = np.zeros(len(class_predictions), dtype=np.float64)
    false_positive = np.zeros(len(class_predictions), dtype=np.float64)

    for index, prediction in enumerate(class_predictions):
        candidates = class_truth.get(prediction["image_id"], [])
        best_iou = 0.0
        best_index = -1
        for candidate_index, candidate in enumerate(candidates):
            if candidate_index in matched[prediction["image_id"]]:
                continue
            iou = box_iou(prediction["box_xyxy"], candidate["box_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_index = candidate_index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched[prediction["image_id"]].add(best_index)
            true_positive[index] = 1.0
        else:
            false_positive[index] = 1.0
    return true_positive, false_positive, truth_count


def detection_metrics(predictions_by_image: dict[str, list[dict]],
                      ground_truth: dict[str, list[dict]]) -> dict[str, float]:
    class_ids = sorted({item["class_id"] for boxes in ground_truth.values() for item in boxes})
    if not class_ids:
        raise ValueError("validation split contains no ground-truth boxes")
    predictions = [
        {**prediction, "image_id": image_id}
        for image_id, image_predictions in predictions_by_image.items()
        for prediction in image_predictions
    ]
    thresholds = np.arange(0.50, 0.96, 0.05)
    aps: dict[float, list[float]] = defaultdict(list)
    total_tp = total_fp = total_truth = 0.0
    for class_id in class_ids:
        for threshold in thresholds:
            tp, fp, truth_count = match_class(predictions, ground_truth, class_id, float(threshold))
            if truth_count == 0:
                continue
            cumulative_tp = np.cumsum(tp)
            cumulative_fp = np.cumsum(fp)
            recall = cumulative_tp / truth_count
            precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1.0e-12)
            aps[round(float(threshold), 2)].append(average_precision(recall, precision))
            if abs(threshold - 0.5) < 1.0e-9:
                total_tp += float(tp.sum())
                total_fp += float(fp.sum())
                total_truth += truth_count

    map_by_iou = {threshold: float(np.mean(values)) for threshold, values in aps.items()}
    return {
        "map50_95": float(np.mean(list(map_by_iou.values()))),
        "map50": map_by_iou[0.5],
        "precision": total_tp / max(total_tp + total_fp, 1.0e-12),
        "recall": total_tp / max(total_truth, 1.0e-12),
    }


def tensor_drift(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"tensor shape mismatch: {reference.shape} vs {candidate.shape}")
    error = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    return {
        "max_abs": float(np.max(error)),
        "mean_abs": float(np.mean(error)),
        "p99_abs": float(np.percentile(error, 99)),
    }


def load_ground_truth(image_path: Path, label_path: Path) -> list[dict]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read validation image: {image_path}")
    return load_yolo_labels(label_path, image.shape[1], image.shape[0])
