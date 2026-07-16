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
