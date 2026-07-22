#!/usr/bin/env python3
"""Load and validate the executable Lesson 12 task-quality contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_QUALITY_CONTRACT = Path(__file__).resolve().parent / "configs/quality_contract.json"
REQUIRED_METRICS = ("map50_95", "map50", "precision", "recall")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_quality_contract(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported quality-contract schema in {path}")
    if not isinstance(document.get("validation_dataset_id"), str):
        raise ValueError("quality contract has no validation_dataset_id")
    if not isinstance(document.get("dataset_manifest_id"), str):
        raise ValueError("quality contract has no dataset_manifest_id")

    shape = document.get("input_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 4
        or shape[0] != 1
        or shape[1] != 3
        or any(not isinstance(value, int) or value <= 0 for value in shape)
    ):
        raise ValueError("quality-contract input_shape must be positive single-image NCHW RGB")

    evaluation = document.get("evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("quality contract has no evaluation settings")
    confidence = evaluation.get("confidence_threshold")
    nms_iou = evaluation.get("nms_iou_threshold")
    max_detections = evaluation.get("max_detections")
    metric = evaluation.get("metric")
    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        raise ValueError("quality-contract confidence_threshold must be in [0, 1]")
    if not isinstance(nms_iou, (int, float)) or not 0.0 <= nms_iou <= 1.0:
        raise ValueError("quality-contract nms_iou_threshold must be in [0, 1]")
    if not isinstance(max_detections, int) or max_detections <= 0:
        raise ValueError("quality-contract max_detections must be positive")
    if not isinstance(metric, str) or not metric:
        raise ValueError("quality contract has no metric implementation ID")

    drops = document.get("maximum_drop_from_pytorch")
    if not isinstance(drops, dict) or set(drops) != set(REQUIRED_METRICS):
        raise ValueError("quality contract must declare every supported regression metric")
    for name, value in drops.items():
        if not isinstance(value, (int, float)) or value < 0.0:
            raise ValueError(f"quality-contract drop for {name} cannot be negative")
    return document


def evaluation_settings(contract: dict[str, Any]) -> dict[str, Any]:
    evaluation = contract["evaluation"]
    return {
        "confidence": float(evaluation["confidence_threshold"]),
        "nms_iou": float(evaluation["nms_iou_threshold"]),
        "max_detections": int(evaluation["max_detections"]),
        "input_shape": list(contract["input_shape"]),
        "metric_implementation": evaluation["metric"],
    }


def regression_thresholds(contract: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(contract["maximum_drop_from_pytorch"][name])
        for name in REQUIRED_METRICS
    }


def contract_identity(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "schema_version": contract["schema_version"],
    }
