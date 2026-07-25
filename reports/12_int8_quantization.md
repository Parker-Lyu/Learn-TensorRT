# Lesson 12 TensorRT 10.14 execution evidence

This report records one complete reproduction of Lesson 12 on 2026-07-25. Generated engines,
calibration caches, raw predictions, and timing samples remain in the lesson's ignored `outputs/`
directory; this concise report preserves the measured decision and enough identity information to
detect accidental comparison across unrelated runs.

## Environment

- GPU: NVIDIA GeForce RTX 4090
- Driver: 595.71.05
- TensorRT: 10.14.1.48
- CUDA Toolkit/runtime: 13.0
- PyTorch: 2.10.0a0+b558c986e8.nv25.11
- ModelOpt: 0.37.0
- Ultralytics: 8.3.225
- Development image: `nvcr.io/nvidia/pytorch:25.11-py3`

## Data qualification

- Calibration: 3,000 fixed COCO train2017 images.
- Validation: all 5,000 labeled COCO val2017 images.
- Dataset manifest: `66f260b27fa20075b701e0b602e11f0098d18bd16e502074bf0d346c23143d77`.
- Calibration selection reproduction: PASS.
- Calibration/validation content overlap check: PASS.
- Representativeness support coverage: PASS.
- Byte-identical preprocessing comparison over all 3,000 calibration images: PASS.

## Detection quality

| Backend | mAP50-95 | mAP50 | Precision | Recall | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| pytorch_fp32 | 0.3631 | 0.5102 | 0.0427 | 0.8098 | PASS |
| pytorch_fp16 | 0.3631 | 0.5104 | 0.0426 | 0.8096 | PASS |
| tensorrt_fp32 | 0.3634 | 0.5106 | 0.0427 | 0.8097 | PASS |
| tensorrt_fp16 | 0.3632 | 0.5105 | 0.0426 | 0.8096 | PASS |
| tensorrt_int8 | 0.3483 | 0.4980 | 0.0409 | 0.8029 | PASS |

The explicit Q/DQ INT8 candidate passed the predeclared PyTorch-FP32-relative and
TensorRT-FP16-relative quality gates. Its mAP50-95 change was
`-0.0149` relative to TensorRT FP16.
The evaluator uses the lesson's fixed COCO-like 101-point contract; it is not a claim of official
`pycocotools` metric equivalence.

## Precision inspection

The Q/DQ ONNX graph contained 131 `QuantizeLinear` and 131 `DequantizeLinear` nodes. TensorRT Engine
Inspector reported `44` INT8,
`50` FP16, and
`2` FP32 compute outputs. It also reported
`41` Q/DQ-origin reformat layers, which is relevant to the measured
performance result.

## Matched performance

Each row uses 500 ms warmup, 120 measured iterations, one inference stream, identical transfer
settings, and engine hashes linked to the quality evaluation.

| Engine | Throughput (qps) | Mean latency (ms) | P50 | P90 | P99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 1164.54 | 1.2266 | 1.1389 | 1.4238 | 2.0096 |
| FP16 | 1969.04 | 0.8728 | 0.8320 | 0.9881 | 1.5087 |
| INT8 | 1590.38 | 0.9897 | 0.9496 | 1.1019 | 1.6427 |

Although Q/DQ INT8 passed the quality gate, FP16 was faster on this RTX 4090: FP16 throughput was
`1969.04` qps versus `1590.38` qps for INT8.
The deployment decision for this measured environment is therefore **retain TensorRT FP16**.

## Legacy entropy-calibrator reference

The isolated `IInt8EntropyCalibrator2` example completed calibration over the same 3,000 images and
was evaluated on the same 5,000-image validation set. It produced mAP50-95 `0.3043` and
mAP50 `0.4333`; its quality gate result was **FAIL**. This supports keeping the
legacy API as reference code rather than the recommended deployment path.

## Reproduction

Follow `12_yolov8_int8_quantization_engineering/docs/reproduction.md`. A different GPU, driver,
model hash, dataset manifest, or TensorRT build requires fresh engines, evaluation, and performance
evidence; the numerical recommendation above must not be copied to an unmatched environment.
