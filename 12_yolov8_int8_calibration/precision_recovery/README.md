# INT8 Precision Recovery Log

This directory records ordered, one-variable-at-a-time experiments for recovering the Lesson 12
INT8 accuracy regression. Generated JSON evidence stays under the lesson's ignored `outputs/`
directory. The fixed COCO validation split and predeclared accuracy thresholds must not change
during these experiments.

## 01 - Preprocessing Parity

Status: **PASS** on 2026-07-16.

The verifier independently calls the production calibration and evaluation preprocessing paths. It
requires exact byte equality after letterbox resize, padding with 114, BGR-to-RGB conversion,
FP32 division by 255, HWC-to-CHW conversion, and contiguous layout. Synthetic tests include odd
dimensions and extreme aspect ratios; the manifest verifier checks the complete hashed calibration
split by default.

Run inside the pinned TensorRT development container from the repository root:

```bash
python3 -m unittest discover \
  -s 12_yolov8_int8_calibration/precision_recovery/01_preprocessing_parity -v
python3 \
  12_yolov8_int8_calibration/precision_recovery/01_preprocessing_parity/verify_preprocessing_parity.py
```

Evidence: `12_yolov8_int8_calibration/outputs/precision_recovery/01_preprocessing_parity.json`.

Recorded result: all 1,000 calibration images produced byte-identical `(1, 3, 640, 640)` FP32
contiguous tensors through both production paths. Zero images failed; 4,915,200,000 tensor bytes
were compared. This rules out preprocessing-path mismatch as the cause of the current INT8 accuracy
drop. The generated JSON also records the dataset manifest and implementation SHA-256 identities.

## Planned Sequence

1. Preprocessing parity.
2. Versioned calibration-set coverage experiment.
3. Entropy versus MinMax calibration.
4. Layer-sensitivity and explicit mixed-precision constraints.
5. Drift-example inspection.
6. QAT only if controlled PTQ experiments remain outside the release gate.
