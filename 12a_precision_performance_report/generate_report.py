#!/usr/bin/env python3
"""Render checkpoint 12a from identity-linked machine-readable lesson artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE_KEYS = {"fp32": "tensorrt_fp32", "fp16": "tensorrt_fp16", "int8": "tensorrt_int8"}


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f(value: float) -> str:
    return f"{value:.3f}"


def validate_evidence(
    performance: dict,
    evaluation: dict,
    diagnosis: dict,
    manifest: dict,
    manifest_sha256: str,
) -> None:
    if performance.get("schema_version") != 3:
        raise ValueError("performance evidence must use schema version 3; recollect lesson 12a")
    if evaluation.get("schema_version") != 1:
        raise ValueError("unsupported lesson 12 evaluation schema")
    if set(performance.get("backends", {})) != set(ENGINE_KEYS):
        raise ValueError("performance evidence must contain fp32, fp16, and int8")
    if set(ENGINE_KEYS.values()) - set(evaluation.get("backends", {})):
        raise ValueError("evaluation evidence is missing required TensorRT backends")

    records = manifest.get("records", [])
    calibration = [record for record in records if record.get("split") == "calibration"]
    validation = [record for record in records if record.get("split") == "validation"]
    if len(calibration) != manifest.get("calibration_count"):
        raise ValueError("manifest calibration_count does not match its records")
    if len(validation) != manifest.get("validation_count"):
        raise ValueError("manifest validation_count does not match its records")
    calibration_hashes = {record["image_sha256"] for record in calibration}
    validation_hashes = {record["image_sha256"] for record in validation}
    if len(calibration_hashes) != len(calibration) or len(validation_hashes) != len(validation):
        raise ValueError("manifest contains duplicate image content within a split")
    if calibration_hashes & validation_hashes:
        raise ValueError("calibration and validation manifests overlap")

    dataset = evaluation.get("dataset", {})
    if dataset.get("dataset_id") != manifest.get("dataset_id"):
        raise ValueError("evaluation dataset_id does not match the selected manifest")
    if dataset.get("validation_images") != len(validation):
        raise ValueError("evaluation image count does not match the selected manifest")
    if dataset.get("manifest_sha256") != manifest_sha256:
        raise ValueError("evaluation manifest SHA-256 does not match the selected manifest")

    for performance_key, evaluation_key in ENGINE_KEYS.items():
        backend = performance["backends"][performance_key]
        if backend.get("sample_count", 0) < 100:
            raise ValueError(f"{performance_key} performance has fewer than 100 samples")
        throughput = backend.get("throughput_qps")
        if (
            not isinstance(throughput, (int, float))
            or isinstance(throughput, bool)
            or not math.isfinite(throughput)
            or throughput <= 0.0
        ):
            raise ValueError(f"{performance_key} performance has invalid trtexec throughput")
        performance_hash = backend.get("engine_sha256")
        evaluation_hash = evaluation["artifacts"][evaluation_key].get("sha256")
        if not performance_hash or performance_hash != evaluation_hash:
            raise ValueError(
                f"{performance_key} engine differs between performance and accuracy evidence"
            )
        drift = evaluation["backends"][evaluation_key].get("tensor_drift_vs_fp32")
        if not isinstance(drift, dict) or set(drift) != {"max_abs", "mean_abs", "p99_abs"}:
            raise ValueError(f"{evaluation_key} is missing raw tensor drift evidence")

    if performance["environment"].get("trtexec") != evaluation["software"].get("tensorrt"):
        raise ValueError("TensorRT version differs between performance and accuracy evidence")
    required_settings = {
        "confidence", "nms_iou", "max_detections", "metric_implementation", "latency_scope"
    }
    if required_settings - set(evaluation.get("settings", {})):
        raise ValueError("evaluation evidence is missing required methodology settings")
    if diagnosis.get("schema_version") != 2:
        raise ValueError("diagnosis evidence must use schema version 2; rerun lesson 11")
    diagnosis["baseline_summary"]["heuristic_diagnosis"]["diagnosis"]
    if diagnosis.get("artifacts", {}).get("engine", {}).get("sha256") != performance[
        "backends"
    ]["fp32"]["engine_sha256"]:
        raise ValueError("lesson 11 diagnosis and lesson 12a performance use different FP32 engines")

    failed = {
        name for name, backend in evaluation["backends"].items()
        if not backend.get("passed", False)
    }
    release = evaluation.get("release_gate", {})
    if set(release.get("failed_backends", [])) != failed:
        raise ValueError("release gate does not agree with backend pass/fail results")
    if bool(release.get("passed")) != (not failed):
        raise ValueError("release gate passed flag is inconsistent")


def backend_decision(performance: dict, evaluation: dict, key: str) -> str:
    label = key.upper()
    mean = performance["backends"][key]["latency_ms"]["mean"]
    baseline = performance["backends"]["fp32"]["latency_ms"]["mean"]
    speed = "faster than" if mean < baseline else "not faster than"
    gate = "passes" if evaluation["backends"][ENGINE_KEYS[key]]["passed"] else "fails"
    return f"{label} is {speed} FP32 and {gate} the predeclared accuracy gate."


def render(
    performance: dict,
    evaluation: dict,
    diagnosis: dict,
    manifest: dict,
    manifest_sha256: str,
) -> str:
    validate_evidence(performance, evaluation, diagnosis, manifest, manifest_sha256)
    records = manifest["records"]
    calibration_count = sum(record["split"] == "calibration" for record in records)
    validation_count = sum(record["split"] == "validation" for record in records)

    accuracy_rows = []
    drift_rows = []
    for key in ("tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8"):
        backend = evaluation["backends"][key]
        metrics = backend["metrics"]
        delta = backend["delta_vs_pytorch"]
        label = key.removeprefix("tensorrt_").upper()
        accuracy_rows.append(
            f"| {label} | {metrics['map50_95']:.4f} | {metrics['map50']:.4f} | "
            f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{delta['map50_95']:+.4f} | {'PASS' if backend['passed'] else 'FAIL'} |"
        )
        drift = backend["tensor_drift_vs_fp32"]
        drift_rows.append(
            f"| {label} | {drift['max_abs']:.6f} | {drift['mean_abs']:.6f} | "
            f"{drift['p99_abs']:.6f} |"
        )

    performance_rows = []
    for key in ("fp32", "fp16", "int8"):
        backend = performance["backends"][key]
        latency = backend["latency_ms"]
        performance_rows.append(
            f"| {key.upper()} | {backend['sample_count']} | {f(latency['mean'])} | "
            f"{f(latency['p50'])} | {f(latency['p90'])} | {f(latency['p99'])} | "
            f"{backend['throughput_qps']:.1f} |"
        )

    release = evaluation["release_gate"]
    dataset = evaluation["dataset"]
    insufficient_evidence = dataset["validation_images"] < 100
    overall = "FAIL" if not release["passed"] or insufficient_evidence else "PASS"
    thresholds = evaluation["regression_thresholds"]
    dominant = diagnosis["baseline_summary"]["heuristic_diagnosis"]["diagnosis"]
    fp16_decision = backend_decision(performance, evaluation, "fp16")
    int8_decision = backend_decision(performance, evaluation, "int8")
    if evaluation["backends"]["tensorrt_int8"]["passed"]:
        recommendation = "INT8 is eligible for deployment consideration under this gate."
    elif evaluation["backends"]["tensorrt_fp16"]["passed"]:
        recommendation = "Retain FP16 while investigating INT8 calibration, mixed precision, or QAT."
    else:
        recommendation = "Neither FP16 nor INT8 is accepted; retain FP32 and investigate drift."

    return f"""# 12a - Precision and Performance Report

Generated from identity-linked JSON artifacts. Overall checkpoint status: **{overall}**.

> Validation evidence: `{dataset['dataset_id']}` contains {dataset['validation_images']} fixed,
> human-labeled images. Dataset manifest SHA-256: `{manifest_sha256}`.

## Environment and Methodology

- GPU/driver/power state: `{performance['environment']['gpu']}`
- TensorRT tool: `{performance['environment']['trtexec']}`
- Warmup: {performance['methodology']['warmup_ms']} ms
- Measured iterations per engine: {performance['methodology']['iterations']}
- Synchronization: {performance['methodology']['synchronization']}
- Accuracy metric: `{evaluation['settings']['metric_implementation']}`
- Detection thresholds: confidence={evaluation['settings']['confidence']}, NMS IoU={evaluation['settings']['nms_iou']}
- Maximum detections per image: {evaluation['settings']['max_detections']}
- Accuracy latency scope: {evaluation['settings']['latency_scope']}
- Calibration/validation overlap: none ({calibration_count} calibration, {validation_count} validation)

## Performance

| Precision | Samples | Mean ms | P50 ms | P90 ms | P99 ms | Throughput (qps) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(performance_rows)}

Latency rows use synchronized `trtexec --exportTimes` measurements. Throughput is the wall-time qps
reported by `trtexec`, which accounts for its transfer/compute overlap. Performance and accuracy
evidence are accepted only when their engine SHA-256 values match.

## Detection Quality and Release Gate

| Precision | mAP50-95 | mAP50 | Precision | Recall | mAP50-95 delta | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(accuracy_rows)}

Predeclared maximum drops: mAP50-95={thresholds['max_map50_95_drop']},
mAP50={thresholds['max_map50_drop']}, precision={thresholds['max_precision_drop']},
recall={thresholds['max_recall_drop']}. Failed backends:
{', '.join(release['failed_backends']) or 'none'}.

{fp16_decision} {int8_decision} {recommendation}

## Raw Tensor Drift Versus TensorRT FP32

| Precision | Max absolute | Mean absolute | P99 absolute |
| --- | ---: | ---: | ---: |
{chr(10).join(drift_rows)}

Drift is diagnostic rather than a release metric. Detection-quality thresholds above control the
decision; high-drift examples in `precision_evaluation.json` identify images for inspection.

## Timeline Diagnosis

Nsight-derived baseline: {dominant} Lesson 17 tests GPU preprocessing as a measured follow-up; the
report does not infer that an optimization worked until new timeline evidence is collected.

## Reproduction

```bash
python3 assets/coco/prepare_coco.py
(cd 11_nsight_performance_diagnosis && python3 profile_yolov8_cpp.py)
(cd 12_yolov8_int8_calibration && python3 build_int8_engine.py --enable-fp16)
(cd 12_yolov8_int8_calibration && python3 compare_engines.py)
python3 12a_precision_performance_report/collect_performance.py
python3 12a_precision_performance_report/generate_report.py
```

The generator rejects mismatched dataset, engine, TensorRT-version, sample-count, drift, and release
gate evidence instead of combining unrelated runs.

## English Summary

This checkpoint compares FP32, FP16, and INT8 YOLOv8n TensorRT engines using matched engine and
dataset identities. {fp16_decision} {int8_decision} {recommendation} The accuracy values use the
documented course COCO-like evaluator, not the official `pycocotools` implementation.

## Three-to-Five-Minute Walkthrough

Explain the dataset and engine identity checks, timing methodology, decoded quality metrics, and raw
tensor drift. State the measured FP16 and INT8 outcomes from the tables, then connect the profiler
diagnosis to the lesson 17 experiment without claiming an unmeasured optimization.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance", type=Path,
                        default=ROOT / "12a_precision_performance_report/outputs/performance.json")
    parser.add_argument("--evaluation", type=Path,
                        default=ROOT / "12_yolov8_int8_calibration/outputs/precision_evaluation.json")
    parser.add_argument("--diagnosis", type=Path,
                        default=ROOT / "11_nsight_performance_diagnosis/outputs/diagnosis_summary.json")
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "assets/coco/data/dataset_manifest.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/12a_precision_performance.md")
    args = parser.parse_args()
    text = render(
        load(args.performance),
        load(args.evaluation),
        load(args.diagnosis),
        load(args.manifest),
        sha256(args.manifest),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
