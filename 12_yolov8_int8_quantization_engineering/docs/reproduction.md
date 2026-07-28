# Lesson 12 reproduction runbook

## 0. Fixed data and preprocessing

Run in `nvcr.io/nvidia/pytorch:25.11-py3` from the repository root:

```bash
python3 assets/coco/prepare_coco.py
python3 12_yolov8_int8_quantization_engineering/tools/prepare_calibration_dataset.py --materialize
python3 12_yolov8_int8_quantization_engineering/tools/analyze_calibration_representativeness.py
python3 12_yolov8_int8_quantization_engineering/tools/verify_preprocessing_parity.py
```

The committed selection and manifest are immutable contracts. A hash mismatch stops the run.
Calibration tensors and evaluation tensors must be contiguous FP32 NCHW RGB, `1x3x640x640`.

## 1. Export the source model

Create `05_torch_to_onnx/outputs/yolov8n.onnx` and its metadata:

```bash
(cd 05_torch_to_onnx && python3 export_yolov8_onnx.py)
python3 05_torch_to_onnx/inspect_onnx.py
python3 05_torch_to_onnx/validate_onnx_runtime.py
```

The ONNX hash is recorded in every engine metadata file. The export script downloads the shared
`assets/yolov8n.pt` weights when they are not already present.

## 2. Evaluate references and build Q/DQ INT8

```bash
python3 12_yolov8_int8_quantization_engineering/modelopt/export_qdq.py \
  --high-precision fp16 --name yolov8n_qdq_fp16
python3 12_yolov8_int8_quantization_engineering/modelopt/build_engines.py
python3 12_yolov8_int8_quantization_engineering/compare_engines.py \
  --experiment-id modelopt_qdq_int8 \
  --fp32-engine 12_yolov8_int8_quantization_engineering/outputs/tensorrt10/references/yolov8n_trt10_fp32.engine \
  --fp16-engine 12_yolov8_int8_quantization_engineering/outputs/tensorrt10/references/yolov8n_trt10_fp16.engine \
  --int8-engine 12_yolov8_int8_quantization_engineering/outputs/tensorrt10/candidate/yolov8n_qdq_int8.engine
python3 12_yolov8_int8_quantization_engineering/modelopt/inspect_precision.py
python3 12_yolov8_int8_quantization_engineering/modelopt/validate_outputs.py
```

The evaluation records PyTorch FP32/FP16, TensorRT FP32/FP16 and Q/DQ INT8 metrics in one report.
Only a candidate with `release_gate.passed=true` is eligible for matched `trtexec` benchmarking.

## 3. Performance evidence

Use `modelopt/benchmark_engines.py` only after the Q/DQ candidate passes the gate. Keep raw
`trtexec` output under `outputs/`; record GPU, driver, CUDA, TensorRT, warmup, iterations, and
transfer settings in the generated JSON.

```bash
python3 12_yolov8_int8_quantization_engineering/modelopt/benchmark_engines.py
```
