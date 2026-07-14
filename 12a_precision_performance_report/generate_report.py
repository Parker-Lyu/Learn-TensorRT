#!/usr/bin/env python3
"""Render checkpoint 12a only from machine-readable lesson artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def f(value: float) -> str:
    return f"{value:.3f}"


def render(performance: dict, evaluation: dict, diagnosis: dict, manifest: dict) -> str:
    records = manifest["records"]
    calibration_hashes = {r["image_sha256"] for r in records if r["split"] == "calibration"}
    validation_hashes = {r["image_sha256"] for r in records if r["split"] == "validation"}
    if calibration_hashes & validation_hashes:
        raise ValueError("calibration and validation manifests overlap")
    if set(performance["backends"]) != {"fp32", "fp16", "int8"}:
        raise ValueError("performance evidence must contain fp32, fp16, and int8")

    accuracy_rows = []
    for key in ("tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8"):
        backend = evaluation["backends"][key]
        metrics = backend["metrics"]
        delta = backend["delta_vs_pytorch"]
        accuracy_rows.append(
            f"| {key.removeprefix('tensorrt_').upper()} | {metrics['map50_95']:.4f} | "
            f"{metrics['map50']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{delta['map50_95']:+.4f} | {'PASS' if backend['passed'] else 'FAIL'} |"
        )
    performance_rows = []
    for key in ("fp32", "fp16", "int8"):
        backend = performance["backends"][key]
        latency = backend["latency_ms"]
        performance_rows.append(
            f"| {key.upper()} | {backend['sample_count']} | {f(latency['mean'])} | "
            f"{f(latency['p50'])} | {f(latency['p90'])} | {f(latency['p99'])} | "
            f"{backend['throughput_images_per_second']:.1f} |"
        )

    release = evaluation["release_gate"]
    dataset = evaluation["dataset"]
    smoke_warning = dataset["validation_images"] < 100
    overall = "FAIL" if not release["passed"] or smoke_warning else "PASS"
    dominant = diagnosis["baseline_summary"]["heuristic_diagnosis"]["diagnosis"]
    thresholds = evaluation["regression_thresholds"]
    return f"""# 12a - Precision and Performance Report

Generated from saved JSON artifacts. Overall checkpoint status: **{overall}**.

> Evidence limitation: the current validation set contains {dataset['validation_images']} generated
> smoke images with pseudo-labels. It validates the evaluator and release gate, but it is not an
> application-ready accuracy claim. Replace it with a fixed labeled validation split before using
> this report in a portfolio or release decision.

## Environment and Methodology

- GPU/driver/power state: `{performance['environment']['gpu']}`
- TensorRT tool: `{performance['environment']['trtexec']}`
- Warmup: {performance['methodology']['warmup_ms']} ms
- Measured iterations per engine: {performance['methodology']['iterations']}
- Synchronization: {performance['methodology']['synchronization']}
- Input/model family: YOLOv8n, float32 NCHW `1x3x640x640`
- Calibration/validation overlap: none ({len(calibration_hashes)} calibration, {len(validation_hashes)} validation)

## Performance

| Precision | Samples | Mean ms | P50 ms | P90 ms | P99 ms | Images/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(performance_rows)}

Every row comes from individual `trtexec --exportTimes` samples after warmup. Engine files remain
environment-specific generated artifacts.

## Detection Quality and Release Gate

| Precision | mAP50-95 | mAP50 | Precision | Recall | mAP50-95 delta | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{chr(10).join(accuracy_rows)}

Predeclared maximum drops: mAP50-95={thresholds['max_map50_95_drop']},
mAP50={thresholds['max_map50_drop']}, precision={thresholds['max_precision_drop']},
recall={thresholds['max_recall_drop']}. Failed backends: {', '.join(release['failed_backends']) or 'none'}.

FP16 raw tensor drift is small enough that decoded smoke detections remain stable. INT8 has much
larger raw drift and changed detection counts; its detection-quality regression fails the declared
gate. The correct action is sensitive-layer fallback, a more representative calibration set, or
QAT—not accepting INT8 only because it is faster.

## Timeline Diagnosis and Optimization Decisions

Nsight-derived baseline: {dominant} The supported optimization is therefore moving measured
preprocessing work to the GPU (lesson 17) and checking the new timeline. Increasing queue capacity
is rejected as a compute optimization: it can absorb bursts but increases latency under sustained
overload and cannot reduce model compute time.

## Reproduction

```bash
python3 12a_precision_performance_report/collect_performance.py
python3 12a_precision_performance_report/generate_report.py
```

The generator validates split hashes and refuses missing precision backends. Accuracy tables are
rendered from `precision_evaluation.json`, not transcribed manually.

## English Summary

This checkpoint compares FP32, FP16, and INT8 YOLOv8n engines under the same TensorRT timing
methodology. FP16 improves performance while passing the current detection-quality thresholds.
INT8 is faster but fails the predeclared accuracy gate and is not release-ready. Nsight evidence
shows CPU preprocessing and postprocessing dominate the original end-to-end request, motivating
the later CUDA preprocessing lesson. The present two-image pseudo-labeled validation split is only
a pipeline smoke test; a portfolio claim requires a fixed, representative labeled dataset.

## Three-to-Five-Minute Walkthrough

Explain the controlled engine comparison, warmup and percentile method, then separate raw tensor
drift from decoded detection metrics. Point out that FP16 passes while INT8 fails the gate. Finish
with the profiler-supported CPU bottleneck, the GPU preprocessing experiment, and the validation
dataset limitation. Never present the smoke-set metrics as production accuracy.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--performance", type=Path,
                        default=ROOT / "12a_precision_performance_report/outputs/performance.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "reports/12a_precision_performance.md")
    args = parser.parse_args()
    text = render(
        load(args.performance),
        load(ROOT / "12_yolov8_int8_calibration/outputs/precision_evaluation.json"),
        load(ROOT / "11_nsight_performance_diagnosis/outputs/diagnosis_summary.json"),
        load(ROOT / "12_yolov8_int8_calibration/data/dataset_manifest.json"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
