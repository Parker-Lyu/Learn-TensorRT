#!/usr/bin/env python3
"""Render a concise Lesson 14 execution summary from generated JSON evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LESSON_DIR = Path(__file__).resolve().parents[1]
OUTPUTS = LESSON_DIR / "outputs"


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing Lesson 14 evidence: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def render(
    parity: dict[str, Any],
    representativeness: dict[str, Any],
    qdq: dict[str, Any],
    evaluation: dict[str, Any],
    audit: dict[str, Any],
    sensitivity: dict[str, Any],
    performance: dict[str, Any],
) -> str:
    if performance.get("schema_version") != 3:
        raise ValueError("performance evidence must use canonical schema version 3")
    release = evaluation.get("release_gate", {})
    int8 = evaluation.get("backends", {}).get("tensorrt_int8", {})
    int8_passed = bool(int8.get("passed"))
    measured = set(performance.get("backends", {}))
    expected = {"fp32", "fp16"} | ({"int8"} if int8_passed else set())
    if measured != expected:
        raise ValueError("performance backends do not follow the INT8 quality-gate policy")

    inspection = qdq.get("onnx_inspection", {})
    counts = audit.get("engines", {}).get("tensorrt_int8", {}).get(
        "compute_output_precision_counts", {}
    )
    metrics = int8.get("metrics", {})
    decision = "ACCEPTED FOR PERFORMANCE COMPARISON" if int8_passed else "REJECTED"
    performance_rows = []
    for name in ("fp32", "fp16", "int8"):
        if name not in performance["backends"]:
            continue
        backend = performance["backends"][name]
        performance_rows.append(
            f"| {name.upper()} | {backend['sample_count']} | "
            f"{backend['latency_ms']['mean']:.3f} | {backend['latency_ms']['p99']:.3f} | "
            f"{backend['throughput_qps']:.1f} |"
        )
    if not int8_passed:
        performance_note = "INT8 was not benchmarked because it failed the predeclared quality gate."
    else:
        performance_note = "INT8 was benchmarked only after it passed the predeclared quality gate."

    return f"""# Lesson 14 Quantization Run Summary

This is a concise execution summary. Lesson 15 validates the linked evidence and produces the
application-facing precision and performance decision report.

## Stage Status

| Stage | Status |
| --- | --- |
| Calibration representativeness | {representativeness['conclusion']['calibration_selection_status']} |
| Preprocessing parity | {parity['status']} |
| Q/DQ ONNX validation | {'PASS' if inspection.get('checker_passed') else 'FAIL'} |
| Unlabeled output sanity | {'PASS' if sensitivity.get('passed') else 'FAIL'} |
| Detection-quality release gate | {'PASS' if release.get('passed') else 'FAIL'} |

## Quantization Evidence

- Dataset: `{evaluation['dataset']['dataset_id']}`
- Validation images: {evaluation['dataset']['validation_images']}
- Calibration images: {qdq['calibration_images']}
- Q/DQ ONNX SHA-256: `{qdq['onnx_sha256']}`
- QuantizeLinear / DequantizeLinear nodes: {inspection.get('quantize_linear_nodes', 0)} / {inspection.get('dequantize_linear_nodes', 0)}
- INT8 engine compute outputs: INT8={counts.get('INT8', 0)}, FP16={counts.get('FP16', 0)}, FP32={counts.get('FP32', 0)}

## INT8 Quality Gate

| mAP50-95 | mAP50 | Precision | Recall | Candidate decision |
| ---: | ---: | ---: | ---: | --- |
| {metrics.get('map50_95', 0.0):.4f} | {metrics.get('map50', 0.0):.4f} | {metrics.get('precision', 0.0):.4f} | {metrics.get('recall', 0.0):.4f} | {decision} |

Failed backends: {', '.join(release.get('failed_backends', [])) or 'none'}.

## Matched Performance Evidence

| Precision | Samples | Mean latency (ms) | P99 latency (ms) | Throughput (qps) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(performance_rows)}

{performance_note}

Detailed generated JSON, logs, timing captures, and inspection artifacts remain under Lesson 14's
ignored `outputs/` directory.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity", type=Path, default=OUTPUTS / "00_qualification/preprocessing_parity.json")
    parser.add_argument(
        "--representativeness", type=Path,
        default=OUTPUTS / "data_preparation/representativeness/representativeness_report.json",
    )
    parser.add_argument("--qdq", type=Path, default=OUTPUTS / "qdq/yolov8n_qdq_fp16.onnx.json")
    parser.add_argument("--evaluation", type=Path, default=OUTPUTS / "evaluation/precision_evaluation.json")
    parser.add_argument("--audit", type=Path, default=OUTPUTS / "tensorrt10/layer_audit.json")
    parser.add_argument("--sensitivity", type=Path, default=OUTPUTS / "tensorrt10/unlabeled_sensitivity.json")
    parser.add_argument(
        "--performance", type=Path,
        default=OUTPUTS / "tensorrt10/performance/performance.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=OUTPUTS / "summary/quantization_run_summary.md",
    )
    args = parser.parse_args()
    text = render(*[
        load(path) for path in (
            args.parity,
            args.representativeness,
            args.qdq,
            args.evaluation,
            args.audit,
            args.sensitivity,
            args.performance,
        )
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Run summary: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
