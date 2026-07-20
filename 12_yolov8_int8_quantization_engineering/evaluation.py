"""Memory-bounded dataset-evaluation helpers shared by lesson 12 runners and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

PREDICTION_DTYPE = np.dtype([
    ("image_index", np.int32),
    ("class_id", np.int16),
    ("confidence", np.float32),
    ("box_xyxy", np.float32, (4,)),
])


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
        normalized = (cx, cy, width, height)
        if (
            class_id < 0
            or not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in normalized)
            or width <= 0.0
            or height <= 0.0
        ):
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


def allocate_prediction_buffer(image_count: int, max_detections: int) -> np.ndarray:
    if image_count <= 0 or max_detections <= 0:
        raise ValueError("image count and maximum detections must be positive")
    return np.empty(image_count * max_detections, dtype=PREDICTION_DTYPE)


def append_predictions(
    buffer: np.ndarray,
    offset: int,
    image_index: int,
    predictions: list[dict],
) -> int:
    end = offset + len(predictions)
    if end > len(buffer):
        raise ValueError("prediction buffer capacity exceeded")
    if not predictions:
        return offset
    view = buffer[offset:end]
    view["image_index"] = image_index
    view["class_id"] = [item["class_id"] for item in predictions]
    view["confidence"] = [item["confidence"] for item in predictions]
    view["box_xyxy"] = [item["box_xyxy"] for item in predictions]
    return end


def match_class(
    class_predictions: np.ndarray,
    class_truth: dict[int, list[dict]],
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    truth_count = sum(len(boxes) for boxes in class_truth.values())
    matched = {
        image_index: np.zeros(len(boxes), dtype=np.bool_)
        for image_index, boxes in class_truth.items()
    }
    true_positive = np.zeros(len(class_predictions), dtype=np.float64)
    false_positive = np.zeros(len(class_predictions), dtype=np.float64)

    for index, prediction in enumerate(class_predictions):
        image_index = int(prediction["image_index"])
        candidates = class_truth.get(image_index, [])
        best_iou = 0.0
        best_index = -1
        for candidate_index, candidate in enumerate(candidates):
            if matched[image_index][candidate_index]:
                continue
            iou = box_iou(prediction["box_xyxy"], candidate["box_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_index = candidate_index
        if best_index >= 0 and best_iou >= iou_threshold:
            matched[image_index][best_index] = True
            true_positive[index] = 1.0
        else:
            false_positive[index] = 1.0
    return true_positive, false_positive, truth_count


def detection_metrics_packed(
    predictions: np.ndarray,
    ground_truth: dict[int, list[dict]],
) -> dict[str, float]:
    truth_class_ids = {
        item["class_id"] for boxes in ground_truth.values() for item in boxes
    }
    if not truth_class_ids:
        raise ValueError("validation split contains no ground-truth boxes")
    predicted_class_ids = set(int(value) for value in np.unique(predictions["class_id"]))
    thresholds = np.arange(0.50, 0.96, 0.05)
    aps: dict[float, list[float]] = {round(float(value), 2): [] for value in thresholds}
    total_tp = total_fp = total_truth = 0.0

    for class_id in sorted(truth_class_ids | predicted_class_ids):
        class_predictions = predictions[predictions["class_id"] == class_id]
        if len(class_predictions):
            order = np.argsort(class_predictions["confidence"])[::-1]
            class_predictions = class_predictions[order]
        class_truth = {
            image_index: [box for box in boxes if box["class_id"] == class_id]
            for image_index, boxes in ground_truth.items()
        }
        class_truth = {key: value for key, value in class_truth.items() if value}
        truth_count = sum(len(boxes) for boxes in class_truth.values())
        if truth_count == 0:
            total_fp += float(len(class_predictions))
            continue
        for threshold in thresholds:
            tp, fp, _ = match_class(class_predictions, class_truth, float(threshold))
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


def detection_metrics(
    predictions_by_image: dict[str, list[dict]],
    ground_truth: dict[str, list[dict]],
) -> dict[str, float]:
    """Compatibility wrapper for focused tests and small callers."""
    image_ids = sorted(set(ground_truth) | set(predictions_by_image))
    image_indexes = {image_id: index for index, image_id in enumerate(image_ids)}
    count = sum(len(items) for items in predictions_by_image.values())
    packed = np.empty(count, dtype=PREDICTION_DTYPE)
    offset = 0
    for image_id, items in predictions_by_image.items():
        offset = append_predictions(packed, offset, image_indexes[image_id], items)
    indexed_truth = {image_indexes[key]: value for key, value in ground_truth.items()}
    return detection_metrics_packed(packed[:offset], indexed_truth)


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

