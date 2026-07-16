# 12a - Precision and Performance Report

This reporting checkpoint combines lessons 06, 11, and 12 without manually copying benchmark
numbers. It collects at least 100 synchronized timing samples for FP32, FP16, and INT8, then renders
performance, raw drift, detection quality, release thresholds, and profiler diagnosis into
`reports/12a_precision_performance.md`.

Run inside the pinned TensorRT container from the repository root:

```bash
python3 assets/coco/prepare_coco.py
(cd 11_nsight_performance_diagnosis && python3 profile_yolov8_cpp.py)
(cd 12_yolov8_int8_calibration && python3 build_int8_engine.py --enable-fp16)
(cd 12_yolov8_int8_calibration && python3 compare_engines.py)
python3 -m unittest discover -s 12a_precision_performance_report/tests -v
python3 12a_precision_performance_report/collect_performance.py
python3 12a_precision_performance_report/generate_report.py
```

The report intentionally fails the overall checkpoint when the INT8 accuracy gate fails or the
validation evidence is not a sufficiently sized fixed labeled dataset. A generated report is not
automatically a passing report.

The generator defaults to `assets/coco/data/dataset_manifest.json`. It rejects evidence unless the
dataset ID, manifest SHA-256, engine SHA-256 values, TensorRT version, minimum sample counts, drift
records, and release-gate state agree across inputs. Keep raw timing JSON and profiler captures in
ignored output directories; commit only the concise report and intentional reproducibility
metadata.

## Precision Decision Policy

The checked-in report currently shows FP32 and FP16 passing while INT8 fails the fixed-dataset
accuracy gate. INT8 is faster, but a speed result cannot override a failed predeclared quality
threshold. FP16 therefore remains the current release candidate.

Implementation experiments for recovering INT8 accuracy belong in lesson 12's
[Accuracy Recovery Workflow](../12_yolov8_int8_calibration/README.md#accuracy-recovery-workflow):
preprocessing parity, a versioned calibration split, calibration-algorithm comparison, layer
sensitivity, explicit mixed-precision constraints, and QAT when PTQ is insufficient. This checkpoint
consumes their evidence; it does not duplicate those implementations.

Change the decision only after all affected evidence is regenerated:

1. build the new engine and calibration cache with distinct artifact names;
2. evaluate PyTorch, FP32, FP16, and the INT8 candidate on the complete unchanged validation split;
3. recollect at least 100 synchronized performance samples for every engine;
4. rerun the profiler evidence when the compared FP32 engine or environment changed;
5. regenerate this report and accept the result only if all identity checks pass.

The generated report is the source of truth for measured numbers and pass/fail state. If a future
candidate passes, update this README summary in the same change so it cannot preserve an obsolete
precision recommendation.
