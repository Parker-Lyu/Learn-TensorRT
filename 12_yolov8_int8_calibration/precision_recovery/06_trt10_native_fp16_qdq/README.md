# Step 06: TensorRT 10 Native FP16 Q/DQ Evidence

This recovery step builds a version-matched TensorRT 10.14 evidence chain for the immutable native
FP16-high-precision ModelOpt Q/DQ graph. Generated engines, logs, timing caches, Inspector JSON, and
reports stay under the ignored `../../outputs/precision_recovery/06_trt10_native_fp16_qdq/` tree.

Status: **QUALITY PASS, DEPLOYMENT RETAINS FP16** on 2026-07-18.

## Fixed experiment

- TensorRT container: `learn-tensorrt-modelopt`, TensorRT `10.14.1.48`.
- FP32 and FP16 references: canonical `05_torch_to_onnx/outputs/yolov8n.onnx`.
- INT8 candidate: Step 05 native FP16 Q/DQ ONNX, without recalibration or re-export.
- Dataset: unchanged coverage manifest with all 5,000 COCO val2017 images.
- Gate: maximum drops of `0.02` mAP50-95, `0.02` mAP50, `0.03` precision, and `0.03` recall.

## Commands

Run every CUDA/TensorRT command inside the persistent TensorRT 10 container:

```bash
docker start learn-tensorrt-modelopt
docker exec learn-tensorrt-modelopt bash -lc '
  cd /workspace/Learn-TensorRT &&
  python3 12_yolov8_int8_calibration/precision_recovery/06_trt10_native_fp16_qdq/\
build_trt10_evidence.py &&
  python3 12_yolov8_int8_calibration/precision_recovery/06_trt10_native_fp16_qdq/\
inspect_trt10_layers.py &&
  python3 12_yolov8_int8_calibration/precision_recovery/06_trt10_native_fp16_qdq/\
validate_trt10_outputs.py
'
```

Only after the unlabeled checks pass, run the one formal labeled gate:

```bash
docker exec learn-tensorrt-modelopt bash -lc '
  cd /workspace/Learn-TensorRT/12_yolov8_int8_calibration &&
  python3 compare_engines.py \
    --manifest outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json \
    --weights ../assets/yolov8n.pt \
    --fp32-engine outputs/precision_recovery/06_trt10_native_fp16_qdq/references/\
yolov8n_trt10_fp32.engine \
    --fp16-engine outputs/precision_recovery/06_trt10_native_fp16_qdq/references/\
yolov8n_trt10_fp16.engine \
    --int8-engine outputs/precision_recovery/06_trt10_native_fp16_qdq/candidate/\
yolov8n_modelopt_hp_fp16_trt10.engine \
    --output-dir outputs/precision_recovery/06_trt10_native_fp16_qdq/evaluation
'
```

Then collect matched performance evidence back-to-back:

```bash
docker exec learn-tensorrt-modelopt bash -lc '
  cd /workspace/Learn-TensorRT &&
  python3 12_yolov8_int8_calibration/precision_recovery/06_trt10_native_fp16_qdq/\
benchmark_trt10_evidence.py
'
```

CPU-focused tests are runnable in the same container:

```bash
python3 -m unittest discover \
  12_yolov8_int8_calibration/precision_recovery/06_trt10_native_fp16_qdq \
  'test_*.py'
```

Quality controls the deployment decision. A failing gate retains FP16. A passing INT8 candidate is
selected only if its matched performance is reproducibly better than the new TensorRT 10 FP16
reference.

## Recorded result

All three optimized engines built successfully with TensorRT `10.14.1.48` and expose the required
FP32 static boundary. Their SHA-256 identities are:

- FP32 reference: `a0bd081a32e6de4554ed182047f9432f0b7aa9481672c56403e30d950e4057fa`;
- FP16 reference: `648995af08b03f4d4355019d357e79615a1060b0ee44f6818acd21f6d411a0c9`;
- native FP16 Q/DQ INT8 candidate:
  `920fbc3ab07b99eab549dc10bdf2a23117beca9bbb0569780705e0e2114365fe`.

The unlabeled eight-image calibration check passed shape, FP32 dtype, finiteness, score range, box
range, and bitwise repeatability checks. Maximum recorded P99 absolute drift versus TensorRT FP32
was `0.1486` for FP16 and `10.3405` for INT8. The formal gate was therefore allowed to proceed;
raw-output drift was not used as a release threshold.

The complete four-backend run processed all 5,000 validation images once and exited with status
`0`:

| Backend | mAP50-95 | mAP50 | Precision | Recall | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| PyTorch | 0.3631 | 0.5102 | 0.0427 | 0.8097 | PASS |
| TensorRT 10 FP32 | 0.3631 | 0.5102 | 0.0427 | 0.8097 | PASS |
| TensorRT 10 FP16 | 0.3635 | 0.5105 | 0.0426 | 0.8096 | PASS |
| TensorRT 10 native Q/DQ INT8+FP16 | 0.3452 | 0.4937 | 0.0440 | 0.8011 | PASS |

The INT8 deltas versus PyTorch were `-0.01789` mAP50-95, `-0.01648` mAP50, `+0.00128`
precision, and `-0.00859` recall. All stayed within the unchanged limits.

Engine Inspector classified the native candidate as 197 total layers with 44 INT8-output compute
layers, 49 FP16-output compute layers, and four FP32-output compute layers. Its 64 INT8-weight
convolutions split into 39 INT8-output and 25 FP16-output convolutions. Native FP16 high-precision
tensors reduced FP32-output compute from the Step 05 count of 12 to four, but format conversion did
not improve: the TensorRT 10 candidate contains 87 reformats, including 41 Q/DQ-origin reformats,
versus 67 total reformats in the TensorRT 8.6 Step 05 engine.

Matched 500 ms warmup and 120-sample `trtexec` evidence was:

| Engine | Mean latency (ms) | P90 (ms) | GPU compute mean (ms) | Throughput (qps) |
| --- | ---: | ---: | ---: | ---: |
| TensorRT 10 FP32 | 5.163 | 5.190 | 3.988 | 248.584 |
| TensorRT 10 FP16 | 2.749 | 2.758 | 1.559 | 635.628 |
| TensorRT 10 native Q/DQ INT8+FP16 | 3.138 | 3.152 | 1.951 | 507.842 |

The native candidate has approximately `20.1%` lower throughput and `25.2%` higher GPU compute
time than the matched FP16 reference. It remains valuable passing quality evidence, but does not
justify the additional Q/DQ complexity or replace FP16 for deployment.

Generated evidence is under
`outputs/precision_recovery/06_trt10_native_fp16_qdq/`, including build metadata, Inspector audit,
unlabeled sensitivity, full accuracy report, process log, and matched performance report. PyTorch
emitted the recorded warning that this CUDA 13.0 build supports the RTX 2060 at its minimum compute
capability and recommends CUDA 12.6/12.8 configurations; no broader compatibility is claimed.
