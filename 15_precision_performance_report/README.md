# 15 - Precision and performance report

## Purpose

This checkpoint consumes Lesson 14's identity-linked accuracy, canonical performance, dataset, and
TensorRT Engine Inspector evidence without repeating the GPU measurement. It validates the evidence
before rendering an application-facing precision and performance decision report.

## Prerequisites

- Complete Lesson 14's reproduction procedure and retain its matched dataset, quality, inspector,
  and canonical performance evidence.
- Use the shared development environment configured in Course 00.

## Deliverables

- `generate_report.py` evidence validator and report generator
- Focused tests for evidence identity, quality-gate, and report-decision behavior
- `reports/15_precision_performance.md` generated decision report

## Decision policy

1. Evaluate PyTorch FP32/FP16 and TensorRT FP32/FP16 on the fixed validation split.
2. Evaluate Q/DQ INT8 with the unchanged quality contract.
3. Benchmark INT8 only after it passes both PyTorch-FP32-relative and TensorRT-FP16-relative gates.
4. Select INT8 only when matched measurements show a meaningful benefit over FP16.
5. Regenerate every affected artifact after a model, dataset, preprocessing, runtime, or engine
   identity changes.

## Generate the Report

```bash
python3 15_precision_performance_report/generate_report.py
```

This validates Lesson 14 evidence identities and renders the application-facing decision report.

Example output (local run):

```text
wrote /workspace/Learn-TensorRT/reports/15_precision_performance.md
```

Complete `14_yolov8_int8_quantization_engineering/docs/reproduction.md` first. The generator rejects
mismatched manifest, engine, runtime, sample, and release-gate identities. A failed INT8 candidate
must have no INT8 performance measurement; the report records the rejection and uses the available
FP32/FP16 performance evidence. The report also consumes Lesson 14's TensorRT 10.14 Engine
Inspector audit. Raw timing captures remain in Lesson 14's ignored output directory.

The generator writes an evidence-backed local report to `reports/15_precision_performance.md`. It
records TensorRT 10.14.1, the matched engine identities, quality results, and the measured
deployment decision. The root `reports/` directory is ignored; regenerate the report whenever the
GPU, driver, model, dataset, or runtime identity differs.

## Outputs

- Canonical raw timing captures and `performance.json` are generated under Lesson 14's ignored
  `outputs/` directory and are not copied into this lesson.
- The generated `reports/15_precision_performance.md` is ignored and environment-specific.

## Tests

Run the Python tests from the repository root:

```bash
python3 -m unittest discover -s 15_precision_performance_report/tests -v
```

The command runs the evidence-identity and release-decision unit tests without requiring a GPU.
<details><summary>Example output (local run)</summary>

```text
test_dataset_identity_mismatch_is_rejected (test_generate_report.GenerateReportTests.test_dataset_identity_mismatch_is_rejected) ... ok
test_engine_identity_mismatch_is_rejected (test_generate_report.GenerateReportTests.test_engine_identity_mismatch_is_rejected) ... ok
test_failed_int8_performance_is_rejected (test_generate_report.GenerateReportTests.test_failed_int8_performance_is_rejected) ... ok
test_invalid_throughput_is_rejected (test_generate_report.GenerateReportTests.test_invalid_throughput_is_rejected) ... ok
test_layer_audit_context_is_rendered (test_generate_report.GenerateReportTests.test_layer_audit_context_is_rendered) ... ok
test_passing_but_slower_int8_retains_fp16 (test_generate_report.GenerateReportTests.test_passing_but_slower_int8_retains_fp16) ... ok
test_performance_must_reference_the_same_evaluation (test_generate_report.GenerateReportTests.test_performance_must_reference_the_same_evaluation) ... ok
test_report_renders_dynamic_decisions_and_drift (test_generate_report.GenerateReportTests.test_report_renders_dynamic_decisions_and_drift) ... ok
test_short_performance_sample_is_rejected (test_generate_report.GenerateReportTests.test_short_performance_sample_is_rejected) ... ok

Ran 9 tests in 0.001s

OK
```
</details>

## Checkpoints

1. Generate a reproducible FP32, FP16, and gated INT8 comparison from machine-readable evidence.
2. Separate raw tensor drift, detection-quality regression, and runtime performance conclusions.
3. Defend the selected deployment precision and rejected alternatives with matched measurements.
