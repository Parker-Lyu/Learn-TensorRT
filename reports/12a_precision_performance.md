# 12a - Precision and Performance Report

Generated from identity-linked JSON artifacts. Overall checkpoint status: **FAIL**.

> Validation evidence: `coco2017-train1000-stratified-v1-seed42-calibration-val5000-human-labels-v1` contains 5000 fixed,
> human-labeled images. Dataset manifest SHA-256: `c7ea8b078c138b68bee1afbf191fcc7b6041b45b38c7906c1d0ac232571dfb3d`.

## Environment and Methodology

- GPU/driver/power state: `NVIDIA GeForce RTX 2060, 580.159.04, P8, [N/A]`
- TensorRT tool: `8.6.1`
- Warmup: 500 ms
- Measured iterations per engine: 120
- Synchronization: trtexec per-inference latency with H2D, compute, and D2H complete
- Accuracy metric: `course-coco-like-101point-v2-no-crowd-no-area-ranges`
- Detection thresholds: confidence=0.001, NMS IoU=0.7
- Maximum detections per image: 300
- Accuracy latency scope: runtime wrapper with H2D, inference, D2H; excludes image loading, preprocessing, and decode
- Calibration/validation overlap: none (1000 calibration, 5000 validation)

## Performance

| Precision | Samples | Mean ms | P50 ms | P90 ms | P99 ms | Throughput (qps) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 120 | 4.409 | 4.408 | 4.432 | 4.442 | 307.2 |
| FP16 | 120 | 2.713 | 2.714 | 2.728 | 2.738 | 652.1 |
| INT8 | 120 | 2.382 | 2.383 | 2.393 | 2.399 | 812.1 |

Latency rows use synchronized `trtexec --exportTimes` measurements. Throughput is the wall-time qps
reported by `trtexec`, which accounts for its transfer/compute overlap. Performance and accuracy
evidence are accepted only when their engine SHA-256 values match.

## Detection Quality and Release Gate

| Precision | mAP50-95 | mAP50 | Precision | Recall | mAP50-95 delta | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP32 | 0.3631 | 0.5102 | 0.0427 | 0.8097 | +0.0000 | PASS |
| FP16 | 0.3631 | 0.5102 | 0.0426 | 0.8095 | +0.0000 | PASS |
| INT8 | 0.3179 | 0.4560 | 0.0429 | 0.7858 | -0.0452 | FAIL |

Predeclared maximum drops: mAP50-95=0.02,
mAP50=0.02, precision=0.03,
recall=0.03. Failed backends:
tensorrt_int8.

FP16 is faster than FP32 and passes the predeclared accuracy gate. INT8 is faster than FP32 and fails the predeclared accuracy gate. Retain FP16 while investigating INT8 calibration, mixed precision, or QAT.

## Raw Tensor Drift Versus TensorRT FP32

| Precision | Max absolute | Mean absolute | P99 absolute |
| --- | ---: | ---: | ---: |
| FP32 | 0.000000 | 0.000000 | 0.000000 |
| FP16 | 41.963776 | 0.009871 | 0.292145 |
| INT8 | 559.510864 | 0.384635 | 10.622131 |

Drift is diagnostic rather than a release metric. Detection-quality thresholds above control the
decision; high-drift examples in `precision_evaluation.json` identify images for inspection.

## Timeline Diagnosis

Nsight-derived baseline: CPU preprocessing and postprocessing dominate the typical measured request. Lesson 17 tests GPU preprocessing as a measured follow-up; the
report does not infer that an optimization worked until new timeline evidence is collected.

## Reproduction

```bash
python3 assets/coco/prepare_coco.py
(cd 11_nsight_performance_diagnosis && python3 profile_yolov8_cpp.py)
(cd 12_yolov8_int8_calibration && python3 build_int8_engine.py --enable-fp16)
(cd 12_yolov8_int8_calibration && python3 compare_engines.py)
python3 12a_precision_performance_report/collect_performance.py
python3 12a_precision_performance_report/generate_report.py
```

The generator rejects mismatched dataset, engine, TensorRT-version, sample-count, drift, and release
gate evidence instead of combining unrelated runs.

## English Summary

This checkpoint compares FP32, FP16, and INT8 YOLOv8n TensorRT engines using matched engine and
dataset identities. FP16 is faster than FP32 and passes the predeclared accuracy gate. INT8 is faster than FP32 and fails the predeclared accuracy gate. Retain FP16 while investigating INT8 calibration, mixed precision, or QAT. The accuracy values use the
documented course COCO-like evaluator, not the official `pycocotools` implementation.

## Three-to-Five-Minute Walkthrough

Explain the dataset and engine identity checks, timing methodology, decoded quality metrics, and raw
tensor drift. State the measured FP16 and INT8 outcomes from the tables, then connect the profiler
diagnosis to the lesson 17 experiment without claiming an unmeasured optimization.
