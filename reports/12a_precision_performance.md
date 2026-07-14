# 12a - Precision and Performance Report

Generated from saved JSON artifacts. Overall checkpoint status: **FAIL**.

> Evidence limitation: the current validation set contains 2 generated
> smoke images with pseudo-labels. It validates the evaluator and release gate, but it is not an
> application-ready accuracy claim. Replace it with a fixed labeled validation split before using
> this report in a portfolio or release decision.

## Environment and Methodology

- GPU/driver/power state: `NVIDIA GeForce RTX 2060, 580.159.04, P8, [N/A]`
- TensorRT tool: `8.6.1`
- Warmup: 500 ms
- Measured iterations per engine: 120
- Synchronization: trtexec per-inference latency with H2D, compute, and D2H complete
- Input/model family: YOLOv8n, float32 NCHW `1x3x640x640`
- Calibration/validation overlap: none (5 calibration, 2 validation)

## Performance

| Precision | Samples | Mean ms | P50 ms | P90 ms | P99 ms | Images/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 120 | 4.421 | 4.416 | 4.455 | 4.486 | 226.2 |
| FP16 | 120 | 2.726 | 2.725 | 2.742 | 2.761 | 366.9 |
| INT8 | 120 | 2.451 | 2.446 | 2.465 | 2.571 | 407.9 |

Every row comes from individual `trtexec --exportTimes` samples after warmup. Engine files remain
environment-specific generated artifacts.

## Detection Quality and Release Gate

| Precision | mAP50-95 | mAP50 | Precision | Recall | mAP50-95 delta | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP32 | 0.4463 | 0.5663 | 0.8750 | 0.7000 | +0.0000 | PASS |
| FP16 | 0.4497 | 0.5663 | 0.8750 | 0.7000 | +0.0034 | PASS |
| INT8 | 0.2296 | 0.3904 | 0.3333 | 0.7000 | -0.2168 | FAIL |

Predeclared maximum drops: mAP50-95=0.02,
mAP50=0.02, precision=0.03,
recall=0.03. Failed backends: tensorrt_int8.

FP16 raw tensor drift is small enough that decoded smoke detections remain stable. INT8 has much
larger raw drift and changed detection counts; its detection-quality regression fails the declared
gate. The correct action is sensitive-layer fallback, a more representative calibration set, or
QAT—not accepting INT8 only because it is faster.

## Timeline Diagnosis and Optimization Decisions

Nsight-derived baseline: CPU preprocessing and postprocessing dominate the typical measured request. The supported optimization is therefore moving measured
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
