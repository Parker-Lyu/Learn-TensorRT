#!/usr/bin/env python3
"""Render lesson 12's report from saved validation artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "12_end_to_end_validation_report" / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the lesson 12 report from saved JSON and log artifacts."
    )
    parser.add_argument(
        "--environment-log",
        type=Path,
        default=OUTPUT_DIR / "environment_check.log",
    )
    parser.add_argument(
        "--pytorch-onnx-report",
        type=Path,
        default=REPO_ROOT / "05_torch_to_onnx" / "outputs" / "validation_report.json",
    )
    parser.add_argument(
        "--precision-report",
        type=Path,
        default=REPO_ROOT / "07_polygraphy_precision_alignment" / "outputs" / "precision_report.json",
    )
    parser.add_argument("--cpp-report", type=Path, default=OUTPUT_DIR / "cpp" / "detections.json")
    parser.add_argument("--cpp-test-log", type=Path, default=OUTPUT_DIR / "cpp_tests.log")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "12_end_to_end_validation.md",
    )
    parser.add_argument("--evidence-output", type=Path, default=OUTPUT_DIR / "evidence.json")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"required JSON artifact was not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return document


def required_path(document: dict[str, Any], key: str, source: str) -> Path:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source} has no usable '{key}' field")
    return Path(value).resolve()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def value(value: Any) -> str:
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def validate(
    pytorch_onnx: dict[str, Any],
    precision: dict[str, Any],
    cpp: dict[str, Any],
    test_log: Path,
) -> dict[str, Any]:
    image = required_path(pytorch_onnx, "image", "PyTorch/ONNX report")
    cpp_image = required_path(cpp, "image", "C++ report")
    if image != cpp_image:
        raise ValueError(
            "controlled-input mismatch: lesson 05 used "
            f"{image}, but lesson 11 used {cpp_image}. Re-run both with --image assets/img.jpeg."
        )

    onnx = required_path(pytorch_onnx, "onnx", "PyTorch/ONNX report")
    if onnx != required_path(precision, "onnx", "precision report"):
        raise ValueError("ONNX artifact mismatch between lesson 05 and lesson 07")

    engine = required_path(precision, "engine", "precision report")
    if engine != required_path(cpp, "engine", "C++ report"):
        raise ValueError("TensorRT engine mismatch between lesson 07 and lesson 11")

    expected_input = (
        REPO_ROOT / "05_torch_to_onnx" / "outputs" / "input_nchw_float32.npy"
    ).resolve()
    if required_path(precision, "input_npy", "precision report") != expected_input:
        raise ValueError("lesson 07 did not use lesson 05's saved NCHW tensor")

    pytorch_metrics = pytorch_onnx.get("comparison")
    trt_comparison = precision.get("comparison")
    if not isinstance(pytorch_metrics, dict):
        raise ValueError("PyTorch/ONNX report has no comparison")
    if not isinstance(trt_comparison, dict) or not isinstance(trt_comparison.get("metrics"), dict):
        raise ValueError("TensorRT alignment is missing; re-run lesson 07 without --skip-trt")

    test_log = test_log.resolve()
    if not test_log.is_file() or "tests passed" not in test_log.read_text(
        encoding="utf-8", errors="replace"
    ).lower():
        raise ValueError(f"C++ test log does not show a passing run: {test_log}")

    samples = cpp.get("latency_samples_ms")
    detections = cpp.get("detections")
    if not isinstance(samples, list) or not samples or not isinstance(detections, list):
        raise ValueError("C++ report is missing measured latency samples or detections")
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError("C++ latency_samples_ms must contain JSON objects")
    expected_count = cpp.get("iterations")
    if expected_count != len(samples):
        raise ValueError("C++ iteration count does not match latency_samples_ms")

    stage_names = list(samples[0])
    if not stage_names or any(set(sample) != set(stage_names) for sample in samples):
        raise ValueError("C++ latency samples do not share the same stages")
    mean_latency = {
        stage: statistics.fmean(float(sample[stage]) for sample in samples)
        for stage in stage_names
    }

    return {
        "image": str(image),
        "onnx": str(onnx),
        "engine": str(engine),
        "input_npy": str(expected_input),
        "pytorch_onnx": pytorch_metrics,
        "onnx_trt": trt_comparison["metrics"],
        "latency_mean_ms": mean_latency,
        "warmup_iterations": cpp.get("warmup_iterations", 0),
        "iterations": cpp.get("iterations", 1),
        "detection_count": len(detections),
        "cpp_test_log": str(test_log),
    }


def metrics_rows(metrics: dict[str, Any], include_shape: bool) -> list[str]:
    rows: list[tuple[str, Any]] = []
    if include_shape:
        rows.append(("Shape", metrics.get("shape")))
    else:
        rows.append(("Shape match", metrics.get("shape_match")))
    rows.extend(
        [
            ("Max absolute error", metrics.get("max_abs_error")),
            ("Mean absolute error", metrics.get("mean_abs_error")),
            ("P99 absolute error", metrics.get("p99_abs_error")),
            ("Tolerance", f"rtol={value(metrics.get('rtol'))}, atol={value(metrics.get('atol'))}"),
            ("Allclose", metrics.get("allclose")),
        ]
    )
    return [f"| {name} | {value(result)} |" for name, result in rows]


def render(evidence: dict[str, Any], environment_log: Path) -> str:
    environment = environment_log.read_text(encoding="utf-8", errors="replace").strip()
    environment = "\n".join(line.rstrip() for line in environment.splitlines())
    if len(environment) > 6000:
        environment = environment[:6000] + "\n... (truncated; see saved log)"
    latency_rows = "\n".join(
        f"| {name} | {value(result)} |" for name, result in evidence["latency_mean_ms"].items()
    )
    pytorch_rows = "\n".join(metrics_rows(evidence["pytorch_onnx"], include_shape=True))
    trt_rows = "\n".join(metrics_rows(evidence["onnx_trt"], include_shape=False))

    return f"""# 12 - End-to-End Validation Report

## Scope

This checkpoint records reproducible evidence for one controlled YOLOv8n input before performance
optimization. It establishes single-input numerical alignment and a working C++ end-to-end path; it
does not establish dataset-level detection accuracy, optimized performance, or service reliability.

## Environment and Dependencies

| Item | Evidence |
| --- | --- |
| Reference environment | TensorRT development container from lesson 00 |
| Environment check | `{relative(environment_log)}` |
| C++ focused tests | Passed; `{relative(Path(evidence['cpp_test_log']))}` |

<details>
<summary>Saved environment-check output</summary>

```text
{environment}
```

</details>

## Controlled Artifacts

| Artifact | Path |
| --- | --- |
| Image | `{relative(Path(evidence['image']))}` |
| ONNX model | `{relative(Path(evidence['onnx']))}` |
| TensorRT engine | `{relative(Path(evidence['engine']))}` |
| NCHW float32 input | `{relative(Path(evidence['input_npy']))}` |

PyTorch, ONNX Runtime, and TensorRT compare the same saved NCHW tensor. The C++ program uses the
same source image and serialized engine.

## Functional Validation

### PyTorch and ONNX Runtime raw output

| Metric | Value |
| --- | ---: |
{pytorch_rows}

### ONNX Runtime and TensorRT raw output

| Metric | Value |
| --- | ---: |
{trt_rows}

### C++ end-to-end smoke test

- Focused preprocessing/postprocessing tests passed.
- C++ inference completed on the controlled image with **{evidence['detection_count']}** detections.
- Machine-readable result: `12_end_to_end_validation_report/outputs/cpp/detections.json`.

## Pipeline Architecture and Ownership

```text
cv::Mat image
  -> preprocess_image (letterbox, BGR->RGB, NCHW float32)
  -> TensorRtRunner
       owns runtime -> engine -> execution context
       owns input/output pinned-host and device buffers
       owns a reusable CUDA stream and timing events
  -> decode_yolov8_output (decode, class-aware NMS, coordinate mapping)
  -> draw_detections and JSON/image reporting
```

`main` owns orchestration and `TensorRtRunner`. The runner uses a private implementation and RAII
wrappers for pinned-host/device CUDA allocations, a reusable stream, and reusable events. It
synchronizes the D2H completion event before decoding host output. Lesson 11 supports one static float32 input and one
float32 output.

## Mean Per-stage Latency Baseline

| Stage | Milliseconds |
| --- | ---: |
{latency_rows}

Each value is the arithmetic mean of {evidence['iterations']} measured samples after
{evidence['warmup_iterations']} warmup iteration(s). The raw samples remain in the C++ JSON result.
Engine deserialization is not included in `total`; this is neither a throughput claim nor an
optimized benchmark. Lesson 13 adds timeline diagnosis.

## What This Evidence Proves

- The documented container workflow can build, test, and run the C++ pipeline.
- PyTorch and ONNX Runtime meet the recorded tolerance for one controlled input.
- ONNX Runtime and TensorRT meet the recorded tolerance for that same input.
- The C++ pipeline produces an annotated image and machine-readable detections.

## What It Does Not Prove Yet

- Dataset-level mAP or detection-quality regression.
- FP16/INT8 acceptance or multi-image accuracy.
- Optimized latency, throughput, concurrency, video behavior, or long-running stability.
- Serialized-engine portability across GPUs, drivers, CUDA versions, or TensorRT versions.

## English Project Summary

I exported YOLOv8n from PyTorch to ONNX, built a TensorRT engine, and implemented an end-to-end C++
inference application. For one controlled image, I compared PyTorch and ONNX Runtime raw outputs,
then compared ONNX Runtime and TensorRT with the same NCHW tensor. The C++ program performs
letterbox preprocessing, TensorRT inference, YOLOv8 decoding, class-aware NMS, coordinate mapping,
visualization, and JSON reporting. This checkpoint proves reproducibility and single-input alignment,
but it is not a dataset-level accuracy or performance certification.

## English Walkthrough (3–5 Minutes)

1. State the deployment goal and identify the controlled image, ONNX model, and TensorRT engine.
2. Explain why raw-output alignment precedes decode and NMS.
3. Walk through the C++ pipeline and ownership boundary inside `TensorRtRunner`.
4. Show the focused tests, annotated image, JSON result, and latency baseline.
5. Close with limitations and the next steps: Nsight profiling in lesson 13, then multi-image
   FP32/FP16/INT8 validation in lesson 14.
"""


def main() -> int:
    args = parse_args()
    try:
        environment_log = args.environment_log.resolve()
        if not environment_log.is_file():
            raise FileNotFoundError(f"environment check log was not found: {environment_log}")
        evidence = validate(
            load_json(args.pytorch_onnx_report),
            load_json(args.precision_report),
            load_json(args.cpp_report),
            args.cpp_test_log,
        )
        args.evidence_output.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render(evidence, environment_log), encoding="utf-8")
        print(f"evidence: {args.evidence_output}")
        print(f"report: {args.output}")
        return 0
    except Exception as error:
        print(f"error: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
