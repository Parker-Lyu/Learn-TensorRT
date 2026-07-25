# 12a - Precision and Performance Report

Generated from identity-linked JSON artifacts. Overall checkpoint status: **PASS**.

> Validation evidence: `coco2017-yolov8n-calibration-v4-val5000-human-labels-v1` contains 5000 fixed,
> human-labeled images. Dataset manifest SHA-256: `66f260b27fa20075b701e0b602e11f0098d18bd16e502074bf0d346c23143d77`.

## Environment and Methodology

- GPU/driver/power state: `NVIDIA GeForce RTX 4090, 595.71.05, P8, 500.00 W`
- TensorRT tool: `10.14.1`
- Warmup: 500 ms
- Measured iterations per engine: 120
- Synchronization: trtexec per-inference latency with H2D, compute, and D2H complete
- Accuracy metric: `course-coco-like-101point-v2-no-crowd-no-area-ranges`
- Detection thresholds: confidence=0.001, NMS IoU=0.7
- Maximum detections per image: 300
- Accuracy latency scope: runtime wrapper with H2D, inference, D2H; excludes image loading, preprocessing, and decode
- Calibration/validation overlap: none (3000 calibration, 5000 validation)

## Performance

| Precision | Samples | Mean ms | P50 ms | P90 ms | P99 ms | Throughput (qps) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 120 | 1.222 | 1.134 | 1.473 | 1.934 | 1172.8 |
| FP16 | 120 | 0.861 | 0.817 | 0.888 | 1.515 | 2001.3 |
| INT8 | 120 | 1.005 | 0.959 | 0.998 | 1.695 | 1578.8 |

Latency rows use synchronized `trtexec --exportTimes` measurements. Throughput is the wall-time qps
reported by `trtexec`, which accounts for its transfer/compute overlap. Performance and accuracy
evidence are accepted only when their engine SHA-256 values match.

## Detection Quality and Release Gate

| Precision | mAP50-95 | mAP50 | Precision | Recall | mAP50-95 delta | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP32 | 0.3634 | 0.5106 | 0.0427 | 0.8097 | +0.0003 | PASS |
| FP16 | 0.3632 | 0.5105 | 0.0426 | 0.8096 | +0.0001 | PASS |
| INT8 | 0.3483 | 0.4980 | 0.0409 | 0.8029 | -0.0148 | PASS |

Predeclared maximum drops: mAP50-95=0.02,
mAP50=0.02, precision=0.03,
recall=0.03. Failed backends:
none.

FP16 is faster than FP32 and passes the predeclared accuracy gate. INT8 is faster than FP32 and passes the predeclared accuracy gate. INT8 passes quality but is slower than matched FP16; retain FP16 for deployment.

## Raw Tensor Drift Versus TensorRT FP32

| Precision | Max absolute | Mean absolute | P99 absolute |
| --- | ---: | ---: | ---: |
| FP32 | 0.000000 | 0.000000 | 0.000000 |
| FP16 | 30.953812 | 0.011010 | 0.295517 |
| INT8 | 545.231018 | 0.488405 | 13.825058 |

Drift is diagnostic rather than a release metric. Detection-quality thresholds above control the
decision; high-drift examples in `precision_evaluation.json` identify images for inspection.

## TensorRT 10.14 layer audit

The TensorRT 10.14 Q/DQ engine contains 44 INT8, 50 FP16, and 2 FP32 compute outputs. This is diagnostic evidence. Use a matched TensorRT 10.14 timeline before attributing an
FP16-versus-Q/DQ difference to a specific layer or runtime cause.

## Reproduction

```bash
# Follow 12_yolov8_int8_quantization_engineering/docs/reproduction.md
python3 12a_precision_performance_report/collect_performance.py
python3 12a_precision_performance_report/generate_report.py
```

The generator rejects mismatched dataset, engine, TensorRT-version, sample-count, drift, and release
gate evidence instead of combining unrelated runs.

## English Summary

This checkpoint compares FP32, FP16, and INT8 YOLOv8n TensorRT engines using matched engine and
dataset identities. FP16 is faster than FP32 and passes the predeclared accuracy gate. INT8 is faster than FP32 and passes the predeclared accuracy gate. INT8 passes quality but is slower than matched FP16; retain FP16 for deployment. The accuracy values use the
documented course COCO-like evaluator, not the official `pycocotools` implementation.

## Three-to-Five-Minute Walkthrough

Explain the dataset and engine identity checks, timing methodology, decoded quality metrics, raw
tensor drift, and TensorRT 10.14 layer audit. State the measured FP16 and INT8 outcomes without
claiming an optimization that was not measured.
