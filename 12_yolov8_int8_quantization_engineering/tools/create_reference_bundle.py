#!/usr/bin/env python3
"""Create an immutable reference bundle from one complete evaluation report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


LESSON_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LESSON_DIR))

from reference_bundle import sha256, write_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--quality-contract", type=Path, default=LESSON_DIR / "configs/quality_contract.json")
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    contract = json.loads(args.quality_contract.read_text(encoding="utf-8"))
    artifacts = report["artifacts"]
    settings = report["settings"]
    evaluation = contract["evaluation"]
    expected_settings = {
        "confidence": evaluation["confidence_threshold"],
        "nms_iou": evaluation["nms_iou_threshold"],
        "max_detections": evaluation["max_detections"],
        "metric_implementation": evaluation["metric"],
        "input_shape": contract["input_shape"],
    }
    changed_settings = [
        name for name, expected in expected_settings.items() if settings.get(name) != expected
    ]
    if changed_settings:
        raise ValueError(
            "reference report does not match the quality contract: "
            + ", ".join(changed_settings)
        )
    drops = contract["maximum_drop_from_pytorch"]
    expected_thresholds = {
        "max_map50_95_drop": drops["map50_95"],
        "max_map50_drop": drops["map50"],
        "max_precision_drop": drops["precision"],
        "max_recall_drop": drops["recall"],
    }
    if report.get("regression_thresholds") != expected_thresholds:
        raise ValueError("reference report thresholds do not match the quality contract")
    identity = {
        "weights_sha256": artifacts["pytorch_weights"]["sha256"],
        "onnx_sha256": sha256(args.onnx),
        "validation_manifest_sha256": report["dataset"]["manifest_sha256"],
        "quality_contract_sha256": sha256(args.quality_contract),
        "preprocessing_id": "yolov8-letterbox-114-rgb-fp32-chw-v1",
        "postprocessing_id": (
            f"yolov8-decode-conf-{settings['confidence']}-nms-{settings['nms_iou']}-"
            f"maxdet-{settings['max_detections']}-v1"
        ),
        "metric_id": settings["metric_implementation"],
        "runtime_id": args.runtime_id,
        "fp32_engine_sha256": artifacts["tensorrt_fp32"]["sha256"],
        "fp16_engine_sha256": artifacts["tensorrt_fp16"]["sha256"],
    }
    bundle = write_bundle(args.output, identity, args.report, report["software"])
    print(f"Reference bundle: {args.output}")
    print(f"Reference ID: {bundle['reference_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
