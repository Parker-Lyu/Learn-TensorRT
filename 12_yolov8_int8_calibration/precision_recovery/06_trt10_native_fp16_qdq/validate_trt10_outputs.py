#!/usr/bin/env python3
"""Run unlabeled calibration-image sanity and raw-output drift checks."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


LESSON_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LESSON_DIR))
from compare_engines import TensorRtRunner  # noqa: E402
from dataset_manifest import load_manifest, resolve_path  # noqa: E402
from evaluation import tensor_drift  # noqa: E402


STEP_OUTPUT = LESSON_DIR / "outputs/precision_recovery/06_trt10_native_fp16_qdq"
DEFAULT_MANIFEST = (
    LESSON_DIR / "outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json"
)
DEFAULT_ENGINES = {
    "tensorrt_fp32": STEP_OUTPUT / "references/yolov8n_trt10_fp32.engine",
    "tensorrt_fp16": STEP_OUTPUT / "references/yolov8n_trt10_fp16.engine",
    "tensorrt_int8": STEP_OUTPUT / "candidate/yolov8n_modelopt_hp_fp16_trt10.engine",
}
EXPECTED_OUTPUT_SHAPE = (1, 84, 8400)


def validate_output(output: np.ndarray) -> dict[str, float | list[int] | str]:
    if output.shape != EXPECTED_OUTPUT_SHAPE:
        raise ValueError(f"unexpected output shape: {output.shape}")
    if output.dtype != np.float32:
        raise ValueError(f"unexpected output dtype: {output.dtype}")
    if not np.isfinite(output).all():
        raise ValueError("output contains NaN or infinity")
    boxes = output[:, :4, :]
    scores = output[:, 4:, :]
    if float(np.max(np.abs(boxes))) > 100000.0:
        raise ValueError("output box coordinates are clearly corrupted")
    if float(np.max(scores)) <= 1e-6:
        raise ValueError("all class scores collapsed near zero")
    if float(np.min(scores)) < -1e-4 or float(np.max(scores)) > 1.0001:
        raise ValueError("class scores fall outside the expected probability range")
    return {
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "min": float(output.min()),
        "max": float(output.max()),
        "box_abs_max": float(np.max(np.abs(boxes))),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
    }


def aggregate_drifts(drifts: list[dict[str, float]]) -> dict[str, float]:
    if not drifts:
        raise ValueError("no drift samples were provided")
    return {
        name: max(item[name] for item in drifts)
        for name in ("max_abs", "mean_abs", "p99_abs")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--images", type=int, default=8)
    parser.add_argument("--fp32-engine", type=Path, default=DEFAULT_ENGINES["tensorrt_fp32"])
    parser.add_argument("--fp16-engine", type=Path, default=DEFAULT_ENGINES["tensorrt_fp16"])
    parser.add_argument("--int8-engine", type=Path, default=DEFAULT_ENGINES["tensorrt_int8"])
    parser.add_argument("--output", type=Path, default=STEP_OUTPUT / "unlabeled_sensitivity.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.images <= 0:
        raise ValueError("--images must be positive")
    engine_paths = {
        "tensorrt_fp32": args.fp32_engine,
        "tensorrt_fp16": args.fp16_engine,
        "tensorrt_int8": args.int8_engine,
    }
    missing = [str(path) for path in [args.manifest, *engine_paths.values()] if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required artifact(s): " + ", ".join(missing))
    manifest = load_manifest(args.manifest)
    records = [record for record in manifest["records"] if record["split"] == "calibration"]
    if len(records) < args.images:
        raise ValueError(f"manifest contains only {len(records)} calibration images")

    runners = {name: TensorRtRunner(path) for name, path in engine_paths.items()}
    drifts = {"tensorrt_fp16": [], "tensorrt_int8": []}
    examples = []
    try:
        for record in records[: args.images]:
            image_path = resolve_path(args.manifest, record["image"])
            outputs = {}
            summaries = {}
            repeatability = {}
            for name, runner in runners.items():
                first = runner.run(image_path, 0.001, 0.7, 300)["output"]
                second = runner.run(image_path, 0.001, 0.7, 300)["output"]
                summaries[name] = validate_output(first)
                repeatability[name] = tensor_drift(first, second)
                if repeatability[name]["max_abs"] != 0.0:
                    raise ValueError(f"{name} is not bitwise deterministic on {record['image']}")
                outputs[name] = first
            for name in drifts:
                drifts[name].append(tensor_drift(outputs["tensorrt_fp32"], outputs[name]))
            examples.append(
                {
                    "image": record["image"],
                    "outputs": summaries,
                    "repeatability": repeatability,
                    "drift_vs_fp32": {name: drifts[name][-1] for name in drifts},
                }
            )
    finally:
        for runner in runners.values():
            runner.close()

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": manifest["dataset_id"],
        "source_split": "calibration",
        "image_count": args.images,
        "passed": True,
        "maximum_drift_vs_fp32": {
            name: aggregate_drifts(values) for name, values in drifts.items()
        },
        "examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["maximum_drift_vs_fp32"], indent=2))
    print(f"Unlabeled sensitivity report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
