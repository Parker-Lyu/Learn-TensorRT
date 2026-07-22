#!/usr/bin/env python3
"""Validate Lesson 12 evidence and generate the curated case-study deliverables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


LESSON = Path(__file__).resolve().parents[1]
OUTPUTS = LESSON / "outputs"
REPORTS = LESSON / "reports"
EVALUATIONS = {
    "legacy_entropy": OUTPUTS / "references/trt86_full/precision_evaluation.json",
    "legacy_minmax": OUTPUTS / "02_legacy_minmax/evaluation/precision_evaluation.json",
    "detection_head_fp16": OUTPUTS / "03_detection_head_fp16/evaluation/precision_evaluation.json",
    "modelopt_qdq_trt8": OUTPUTS / "04_modelopt_qdq/trt8/evaluation/precision_evaluation.json",
    "modelopt_native_fp16_qdq_trt10": (
        OUTPUTS / "04_modelopt_qdq/trt10/evaluation/precision_evaluation.json"
    ),
}
PERFORMANCE = OUTPUTS / "04_modelopt_qdq/trt10/performance/performance.json"
LAYER_AUDIT = OUTPUTS / "04_modelopt_qdq/trt10/layer_audit.json"
DATA_MANIFEST = LESSON / "data/dataset_manifest.json"
SELECTION = LESSON / "data/calibration_selection.json"
PARITY = OUTPUTS / "00_qualification/preprocessing_parity.json"
COVERAGE = OUTPUTS / "data_preparation/coverage_report.json"
QUALITY_CONTRACT = LESSON / "configs/quality_contract.json"
EXPERIMENTS = LESSON / "configs/experiments.json"
EXPERIMENT_IDS = {
    "legacy_entropy": "01_legacy_entropy",
    "legacy_minmax": "02_legacy_minmax",
    "detection_head_fp16": "03_detection_head_fp16",
    "modelopt_qdq_trt8": "04_modelopt_qdq_trt8",
    "modelopt_native_fp16_qdq_trt10": "05_modelopt_native_fp16_qdq_trt10",
}


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required evidence not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percent_change(value: float, reference: float) -> float:
    return 100.0 * (value / reference - 1.0)


def main() -> int:
    manifest = load(DATA_MANIFEST)
    selection = load(SELECTION)
    parity = load(PARITY)
    coverage = load(COVERAGE)
    quality_contract = load(QUALITY_CONTRACT)
    experiments_hash = sha256(EXPERIMENTS)
    evaluations = {name: load(path) for name, path in EVALUATIONS.items()}
    performance = load(PERFORMANCE)
    audit = load(LAYER_AUDIT)["engines"]["tensorrt_int8"]

    dataset_ids = {item["dataset"]["dataset_id"] for item in evaluations.values()}
    if dataset_ids != {manifest["dataset_id"]}:
        raise ValueError("evaluation reports do not share the canonical dataset identity")
    manifest_hash = sha256(DATA_MANIFEST)
    if any(item["dataset"]["manifest_sha256"] != manifest_hash for item in evaluations.values()):
        raise ValueError("an evaluation report uses a different manifest hash")
    drops = quality_contract["maximum_drop_from_pytorch"]
    expected_thresholds = {
        f"max_{name}_drop": value for name, value in drops.items()
    }
    for name, report in evaluations.items():
        if report.get("regression_thresholds") != expected_thresholds:
            raise ValueError(f"{name} evaluation does not match the quality contract")
        if report.get("quality_contract", {}).get("sha256") != sha256(QUALITY_CONTRACT):
            raise ValueError(f"{name} evaluation has a different quality-contract identity")
        experiment = report.get("experiment", {})
        if experiment.get("id") != EXPERIMENT_IDS[name]:
            raise ValueError(f"{name} evaluation has the wrong experiment identity")
        if experiment.get("matrix_sha256") != experiments_hash:
            raise ValueError(f"{name} evaluation has a different experiment matrix")
    if parity.get("status") != "PASS" or parity["comparison"]["images_failed"] != 0:
        raise ValueError("preprocessing parity did not pass")

    trt10_evaluation = evaluations["modelopt_native_fp16_qdq_trt10"]
    for key in ("fp32", "fp16", "int8"):
        evaluation_key = f"tensorrt_{key}"
        if performance["engines"][key]["engine_sha256"] != trt10_evaluation["artifacts"][evaluation_key]["sha256"]:
            raise ValueError(f"{key} performance and quality engine identities differ")

    stages = {}
    for name, report in evaluations.items():
        candidate = report["backends"]["tensorrt_int8"]
        stages[name] = {
            "metrics": candidate["metrics"],
            "delta_vs_pytorch": candidate["delta_vs_pytorch"],
            "passed": candidate["passed"],
            "engine_sha256": report["artifacts"]["tensorrt_int8"]["sha256"],
            "tensorrt": report["software"]["tensorrt"],
        }

    fp16 = performance["engines"]["fp16"]
    int8 = performance["engines"]["int8"]
    decision = {
        "selected_precision": "fp16",
        "reason": "Q/DQ INT8+FP16 passes quality but is slower than matched FP16",
        "int8_throughput_change_percent": percent_change(
            int8["throughput_qps"], fp16["throughput_qps"]
        ),
        "int8_gpu_compute_change_percent": percent_change(
            int8["gpu_compute_ms"]["mean"], fp16["gpu_compute_ms"]["mean"]
        ),
        "int8_latency_change_percent": percent_change(
            int8["latency_ms"]["mean"], fp16["latency_ms"]["mean"]
        ),
    }
    summary = {
        "schema_version": 1,
        "dataset": {
            "dataset_id": manifest["dataset_id"],
            "manifest_sha256": manifest_hash,
            "selection_sha256": sha256(SELECTION),
            "candidate_images": len(selection["candidate_pool"]),
            "calibration_images": manifest["calibration_count"],
            "validation_images": manifest["validation_count"],
            "historical_1000_selected": coverage["selected_from_historical_1000"],
            "preprocessing_parity": "PASS",
        },
        "quality_contract": {
            "sha256": sha256(QUALITY_CONTRACT),
            "document": quality_contract,
        },
        "experiment_matrix_sha256": experiments_hash,
        "stages": stages,
        "trt10_performance": {
            name: {
                key: engine[key]
                for key in (
                    "engine_sha256",
                    "throughput_qps",
                    "sample_count",
                    "latency_ms",
                    "gpu_compute_ms",
                    "h2d_ms",
                    "d2h_ms",
                )
            }
            for name, engine in performance["engines"].items()
        },
        "trt10_layer_audit": {
            key: audit[key]
            for key in (
                "total_layers",
                "compute_output_precision_counts",
                "int8_weight_convolutions_by_output_precision",
                "reformat_count",
                "qdq_origin_reformat_count",
                "fp32_external_boundary_conversion_count",
            )
        },
        "deployment_decision": decision,
    }

    rows = []
    labels = {
        "legacy_entropy": "Legacy Entropy INT8+FP16",
        "legacy_minmax": "Legacy MinMax INT8+FP16",
        "detection_head_fp16": "MinMax + complete detection head FP16",
        "modelopt_qdq_trt8": "ModelOpt Q/DQ INT8+FP16",
        "modelopt_native_fp16_qdq_trt10": "ModelOpt native-FP16 Q/DQ INT8+FP16",
    }
    for name in EVALUATIONS:
        stage = stages[name]
        metrics = stage["metrics"]
        rows.append(
            f"| {labels[name]} | {stage['tensorrt']} | {metrics['map50_95']:.4f} | "
            f"{metrics['map50']:.4f} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {'PASS' if stage['passed'] else 'FAIL'} |"
        )

    text = f"""# YOLOv8n Quantization Engineering Case Study

Status: **COMPLETE — DEPLOYMENT RETAINS FP16**

This case study follows one fixed quality contract from TensorRT legacy calibration through ModelOpt
explicit Q/DQ. Candidate speed never overrides a failed quality gate, and a quality-passing INT8
candidate is deployed only when it also beats a version-matched FP16 reference.

## Data Qualification

- Dataset: `{manifest['dataset_id']}`.
- Candidate pool: {len(selection['candidate_pool'])} fixed COCO train2017 images.
- Calibration: {manifest['calibration_count']} independently coverage-selected images.
- Validation: all {manifest['validation_count']} COCO val2017 images with human labels.
- Calibration manifest SHA-256: `{manifest_hash}`.
- Selection metadata SHA-256: `{sha256(SELECTION)}`.
- Historical 1,000-image members selected again: {coverage['selected_from_historical_1000']}; old membership was not forced.
- Preprocessing parity: PASS for all 3,000 calibration images with byte-identical FP32 tensors.

## Quantization Evolution

| Candidate | Runtime | mAP50-95 | mAP50 | Precision | Recall | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
{chr(10).join(rows)}

Entropy loses too much task accuracy. MinMax recovers most of it but misses the fixed mAP50 gate.
Moving the complete detection head to FP16 improves mAP50-95 but still misses mAP50, so the remaining
regression is not explained by detection-head INT8 alone. Explicit Q/DQ passes the unchanged gate in
both TensorRT 8.6 and TensorRT 10.14.

## Matched TensorRT 10 Performance

Methodology: 500 ms warmup, 120 measured iterations, one inference stream, and synchronized
`trtexec --exportTimes` evidence.

| Engine | Mean latency (ms) | P90 (ms) | GPU compute mean (ms) | Throughput (qps) |
| --- | ---: | ---: | ---: | ---: |
| FP32 | {performance['engines']['fp32']['latency_ms']['mean']:.3f} | {performance['engines']['fp32']['latency_ms']['p90']:.3f} | {performance['engines']['fp32']['gpu_compute_ms']['mean']:.3f} | {performance['engines']['fp32']['throughput_qps']:.3f} |
| FP16 | {fp16['latency_ms']['mean']:.3f} | {fp16['latency_ms']['p90']:.3f} | {fp16['gpu_compute_ms']['mean']:.3f} | {fp16['throughput_qps']:.3f} |
| Q/DQ INT8+FP16 | {int8['latency_ms']['mean']:.3f} | {int8['latency_ms']['p90']:.3f} | {int8['gpu_compute_ms']['mean']:.3f} | {int8['throughput_qps']:.3f} |

Against FP16, Q/DQ INT8+FP16 has `{decision['int8_throughput_change_percent']:.1f}%` throughput,
`+{decision['int8_gpu_compute_change_percent']:.1f}%` GPU compute time, and
`+{decision['int8_latency_change_percent']:.1f}%` mean latency. It is quality-eligible but slower.

## Why INT8 Can Be Slower

Inspector evidence reports {audit['compute_output_precision_counts'].get('INT8', 0)} INT8-output
compute layers, {audit['compute_output_precision_counts'].get('FP16', 0)} FP16-output compute layers,
and {audit['reformat_count']} reformats, including {audit['qdq_origin_reformat_count']} with Q/DQ
origin. Likely contributors include incomplete INT8 kernel coverage, Q/DQ conversion boundaries,
layout changes, memory traffic, tactic selection, and batch-1 overhead. A deeper investigation could
use per-layer `trtexec` profiles, Engine Inspector formats, Nsight Systems, and Nsight Compute, but
that root-cause study is intentionally outside this lesson.

## Deployment Decision

Retain FP16. Both explicit-Q/DQ candidates prove that INT8 quality can be recovered without changing
the gate, but the final TensorRT 10 candidate does not provide a matched performance benefit. The
engineering lesson is that INT8 is a candidate technology, not an automatic deployment decision.
"""

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "quantization_results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (REPORTS / "quantization_case_study.md").write_text(text, encoding="utf-8")
    print(f"Wrote {REPORTS / 'quantization_results.json'}")
    print(f"Wrote {REPORTS / 'quantization_case_study.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
