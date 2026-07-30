# 15 - Precision and performance report

## Purpose

This checkpoint consumes Lesson 14's TensorRT 10.14 FP32, FP16, and quality-passing Q/DQ INT8
engines. It records at least 100 synchronized `trtexec` samples, wall-time throughput, detection
metrics, release-gate state, engine hashes, dataset identity, and TensorRT 10.14 Engine Inspector
evidence without copying numbers by
hand.

Run in the pinned course container from the repository root:

## Prerequisites

- Complete lesson 14's reproduction procedure and retain its matched engine, dataset, quality, and inspector evidence.
- Use the same pinned GPU environment for all compared precision modes.

## Deliverables

- `collect_performance.py` evidence collector
- `generate_report.py` report generator and focused tests
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
python3 15_precision_performance_report/collect_performance.py
python3 15_precision_performance_report/generate_report.py
```

Complete `14_yolov8_int8_quantization_engineering/docs/reproduction.md` first. The generator rejects
an INT8 candidate that failed its gate and rejects mismatched manifest, engine, runtime, or sample
identities. The report also consumes Lesson 14's TensorRT 10.14 Engine Inspector audit. Raw timing
captures remain in ignored output directories.

The generator writes an evidence-backed local report to `reports/15_precision_performance.md`. It
records TensorRT 10.14.1, the matched engine identities, quality results, and the measured
deployment decision. The root `reports/` directory is ignored; regenerate the report whenever the
GPU, driver, model, dataset, or runtime identity differs.

## Outputs

- Raw timing captures and `performance.json` are written under ignored `outputs/`.
- The generated `reports/15_precision_performance.md` is ignored and environment-specific.

## Tests

Run the Python tests from the repository root:

```bash
python3 -m unittest discover -s 15_precision_performance_report/tests -v
```

## Checkpoints

1. Generate a reproducible FP32, FP16, and gated INT8 comparison from machine-readable evidence.
2. Separate raw tensor drift, detection-quality regression, and runtime performance conclusions.
3. Defend the selected deployment precision and rejected alternatives with matched measurements.
