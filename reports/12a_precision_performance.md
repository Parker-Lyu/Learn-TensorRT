# 12a - Precision and Performance Report

Generated from identity-linked JSON artifacts. Overall checkpoint status: **PASS**.

> Validation evidence: `coco2017-yolov8n-calibration-v3-val5000-human-labels-v1` contains 5000 fixed,
> human-labeled images. Dataset manifest SHA-256: `38c88bff89757dba6e22c44d30398ae0d17f8bd11ec2c09a867b3e975d339a50`.

## Environment and Methodology

- GPU/driver/power state: `NVIDIA GeForce RTX 2060, 580.159.04, P8, [N/A]`
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
| FP32 | 120 | 5.161 | 5.160 | 5.190 | 5.214 | 249.2 |
| FP16 | 120 | 2.739 | 2.739 | 2.754 | 2.769 | 640.8 |
| INT8 | 120 | 3.086 | 3.085 | 3.100 | 3.111 | 522.5 |

Latency rows use synchronized `trtexec --exportTimes` measurements. Throughput is the wall-time qps
reported by `trtexec`, which accounts for its transfer/compute overlap. Performance and accuracy
evidence are accepted only when their engine SHA-256 values match.

## Detection Quality and Release Gate

| Precision | mAP50-95 | mAP50 | Precision | Recall | mAP50-95 delta | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP32 | 0.3631 | 0.5102 | 0.0427 | 0.8097 | -0.0000 | PASS |
| FP16 | 0.3630 | 0.5104 | 0.0426 | 0.8098 | -0.0001 | PASS |
| INT8 | 0.3454 | 0.4946 | 0.0459 | 0.7998 | -0.0177 | PASS |

Predeclared maximum drops: mAP50-95=0.02,
mAP50=0.02, precision=0.03,
recall=0.03. Failed backends:
none.

FP16 is faster than FP32 and passes the predeclared accuracy gate. INT8 is faster than FP32 and passes the predeclared accuracy gate. INT8 passes quality but is slower than matched FP16; retain FP16 for deployment.

## Raw Tensor Drift Versus TensorRT FP32

| Precision | Max absolute | Mean absolute | P99 absolute |
| --- | ---: | ---: | ---: |
| FP32 | 0.000000 | 0.000000 | 0.000000 |
| FP16 | 37.226578 | 0.009605 | 0.286224 |
| INT8 | 535.205200 | 0.441170 | 12.536011 |

Drift is diagnostic rather than a release metric. Detection-quality thresholds above control the
decision; high-drift examples in `precision_evaluation.json` identify images for inspection.

## Timeline Diagnosis

Lesson 11 historical Nsight baseline: CPU preprocessing and postprocessing dominate the typical measured request. This diagnosis is contextual evidence from the
pinned TensorRT 8.6 pipeline and is not identity-linked to the TensorRT 10 precision engines. A new
timeline capture is required before attributing the TensorRT 10 FP16-versus-Q/DQ difference to a
specific layer or runtime cause.

## Reproduction

```bash
# Follow 12_yolov8_int8_quantization_engineering/docs/reproduction.md
python3 12_yolov8_int8_quantization_engineering/tools/generate_case_study.py
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

Explain the dataset and engine identity checks, timing methodology, decoded quality metrics, and raw
tensor drift. State the measured FP16 and INT8 outcomes from the tables, then connect the profiler
diagnosis to the lesson 17 experiment without claiming an unmeasured optimization.
