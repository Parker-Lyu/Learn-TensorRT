# 12a - Precision and Performance Report

This reporting checkpoint combines lessons 06, 11, and 12 without manually copying benchmark
numbers. It collects at least 100 synchronized timing samples for FP32, FP16, and INT8, then renders
performance, raw drift, detection quality, release thresholds, and profiler diagnosis into
`reports/12a_precision_performance.md`.

Run inside the pinned TensorRT container from the repository root:

```bash
python3 -m unittest discover -s 12a_precision_performance_report/tests -v
python3 12a_precision_performance_report/collect_performance.py
python3 12a_precision_performance_report/generate_report.py
```

The report intentionally fails the overall checkpoint when the INT8 accuracy gate fails or the
validation dataset is only the generated smoke fixture. A generated report is not automatically a
passing report.

Before portfolio use, replace lesson 12's two-image pseudo-labeled fixture with a fixed,
representative labeled validation split, rerun every backend with identical evaluator settings,
and regenerate the report. Keep raw timing JSON and profiler captures in ignored output directories;
commit the concise report and intentional manifests.
