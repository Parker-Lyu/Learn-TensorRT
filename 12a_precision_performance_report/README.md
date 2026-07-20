# 12a - Precision and Performance Report

This reporting checkpoint combines lessons 06, 11, and the formal quantization case study without manually copying benchmark
numbers. It collects at least 100 synchronized timing samples for FP32, FP16, and INT8, records the
wall-time throughput reported by `trtexec`, then renders performance, raw drift, detection quality,
release thresholds, and profiler diagnosis into
`reports/12a_precision_performance.md`.

Latency percentiles come from `trtexec --exportTimes`. Throughput comes from the `trtexec` summary
rather than `1000 / mean latency`, because transfers and compute from different inferences may
overlap.

Run inside the pinned TensorRT container from the repository root:

```bash
(cd 11_nsight_performance_diagnosis && python3 profile_yolov8_cpp.py)
python3 -m unittest discover -s 12a_precision_performance_report/tests -v
python3 12a_precision_performance_report/collect_performance.py
python3 12a_precision_performance_report/generate_report.py
```

Before these checkpoint commands, complete the runbook in the new Lesson 12 README and select its
quality-passing final candidate.

The report intentionally fails the overall checkpoint when the INT8 accuracy gate fails or the
validation evidence is not a sufficiently sized fixed labeled dataset. A generated report is not
automatically a passing report.

The generator consumes
`12_yolov8_int8_quantization_engineering/data/dataset_manifest.json`. It rejects evidence unless the
dataset ID, manifest SHA-256, engine SHA-256 values, TensorRT version, minimum sample counts, drift
records, and release-gate state agree across inputs. Keep raw timing JSON and profiler captures in
ignored output directories; commit only the concise report and intentional reproducibility
metadata.

## Precision Decision Policy

The current report uses the reorganized Lesson 12 evidence. Explicit Q/DQ INT8+FP16 passes every
fixed quality threshold, but matched TensorRT 10 throughput is `522.188 qps` versus `636.729 qps`
for FP16. FP16 therefore remains the deployment choice because the quality-passing INT8 candidate
does not provide a performance benefit.

Quantization implementation and candidate selection belong in the new Lesson 12. This checkpoint
consumes only its final quality-eligible candidate; it does not repeat failed-candidate experiments
or the optional root-cause investigation for slower INT8+FP16 execution.

Change the decision only after all affected evidence is regenerated:

1. build the new engine and calibration cache with distinct artifact names;
2. evaluate PyTorch, FP32, FP16, and the INT8 candidate on the complete unchanged validation split;
3. recollect at least 100 synchronized performance samples for every engine;
4. rerun the profiler evidence when the compared FP32 engine or environment changed;
5. regenerate this report and accept the result only if all identity checks pass.

The generated report is the source of truth for measured numbers and pass/fail state. If a future
candidate passes, update this README summary in the same change so it cannot preserve an obsolete
precision recommendation.
