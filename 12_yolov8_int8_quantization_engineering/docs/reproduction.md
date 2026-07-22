# Reproduction Runbook

Run GPU and TensorRT commands only in the matching containers declared in
`../configs/environments.json`.

## 0. Reproduce The Fixed Dataset

Run from the repository root inside `trt_dev`. The first command prepares the pinned COCO
annotations and labeled val2017 split. The second downloads the exact committed 5,000-image
candidate pool, verifies every image hash, recomputes the ordered 3,000-image selection, and
materializes it for calibration:

```bash
python3 assets/coco/prepare_coco.py

python3 12_yolov8_int8_quantization_engineering/tools/prepare_calibration_dataset.py \
  --download-candidates \
  --materialize
```

`data/calibration_selection.json` is a read-only release contract, not a generated local baseline.
The preparation command must reproduce it exactly. Regenerated selection metadata and coverage
evidence are written below `outputs/data_preparation/`; a mismatch stops preparation without
changing the committed contract.

## 1. TensorRT 8.6 Reference And Candidates

The first TensorRT 8.6 full evaluation establishes the reference and evaluates Entropy in the same
5,000-image pass. Later TRT8 candidates reuse that report and run only their new engine.

Run inside `trt_dev` from the new lesson directory:

```bash
cd /workspace/Projects/Learn-TensorRT/12_yolov8_int8_quantization_engineering

python3 build_int8_engine.py \
  --calibrator entropy \
  --enable-fp16 \
  --timing-cache outputs/trt86.timing.cache \
  --output outputs/01_legacy_entropy/yolov8n_entropy.engine \
  --cache outputs/01_legacy_entropy/yolov8n_entropy.cache

python3 compare_engines.py \
  --experiment-id 01_legacy_entropy \
  --int8-engine outputs/01_legacy_entropy/yolov8n_entropy.engine \
  --output-dir outputs/references/trt86_full

python3 tools/create_reference_bundle.py \
  --report outputs/references/trt86_full/precision_evaluation.json \
  --onnx ../05_torch_to_onnx/outputs/yolov8n.onnx \
  --output outputs/references/trt86_full/reference_bundle.json

python3 build_int8_engine.py \
  --calibrator minmax \
  --enable-fp16 \
  --timing-cache outputs/trt86.timing.cache \
  --output outputs/02_legacy_minmax/yolov8n_minmax.engine \
  --cache outputs/02_legacy_minmax/yolov8n_minmax.cache

python3 compare_engines.py \
  --experiment-id 02_legacy_minmax \
  --reference-bundle outputs/references/trt86_full/reference_bundle.json \
  --int8-engine outputs/02_legacy_minmax/yolov8n_minmax.engine \
  --output-dir outputs/02_legacy_minmax/evaluation

python3 build_int8_engine.py \
  --calibrator minmax \
  --precision-profile detection_head_fp16 \
  --enable-fp16 \
  --timing-cache outputs/trt86.timing.cache \
  --output outputs/03_detection_head_fp16/yolov8n_minmax_head_fp16.engine \
  --cache outputs/02_legacy_minmax/yolov8n_minmax.cache

python3 compare_engines.py \
  --experiment-id 03_detection_head_fp16 \
  --reference-bundle outputs/references/trt86_full/reference_bundle.json \
  --int8-engine outputs/03_detection_head_fp16/yolov8n_minmax_head_fp16.engine \
  --output-dir outputs/03_detection_head_fp16/evaluation
```

Export Q/DQ INT8 graphs inside `learn-tensorrt-modelopt`. FP32 high-precision tensors target
TensorRT 8.6; native FP16 high-precision tensors target the TensorRT 10 INT8+FP16 candidate:

```bash
cd /workspace/Learn-TensorRT

python3 12_yolov8_int8_quantization_engineering/modelopt/modelopt_ptq.py \
  --high-precision fp32 \
  --name yolov8n_modelopt_qdq_calibration_v3

python3 12_yolov8_int8_quantization_engineering/modelopt/modelopt_ptq.py \
  --high-precision fp16 \
  --name yolov8n_modelopt_qdq_native_fp16_calibration_v3
```

Return to `trt_dev` to build and evaluate the TRT8 Q/DQ candidate against the existing TRT8
reference. Then use the ModelOpt container to build a complete, version-matched TRT10 evidence set:

```bash
# trt_dev
cd /workspace/Projects/Learn-TensorRT
python3 12_yolov8_int8_quantization_engineering/modelopt/build_trt86_qdq_engine.py

cd 12_yolov8_int8_quantization_engineering
python3 compare_engines.py \
  --experiment-id 04_modelopt_qdq_trt8 \
  --reference-bundle outputs/references/trt86_full/reference_bundle.json \
  --int8-engine outputs/04_modelopt_qdq/trt8/yolov8n_modelopt_qdq_trt86_int8_fp16.engine \
  --output-dir outputs/04_modelopt_qdq/trt8/evaluation

# learn-tensorrt-modelopt
cd /workspace/Learn-TensorRT
python3 12_yolov8_int8_quantization_engineering/modelopt/build_trt10_evidence.py
python3 12_yolov8_int8_quantization_engineering/modelopt/inspect_trt10_layers.py
python3 12_yolov8_int8_quantization_engineering/modelopt/validate_trt10_outputs.py

cd 12_yolov8_int8_quantization_engineering
python3 compare_engines.py \
  --experiment-id 05_modelopt_native_fp16_qdq_trt10 \
  --fp32-engine outputs/04_modelopt_qdq/trt10/references/yolov8n_trt10_fp32.engine \
  --fp16-engine outputs/04_modelopt_qdq/trt10/references/yolov8n_trt10_fp16.engine \
  --int8-engine outputs/04_modelopt_qdq/trt10/candidate/yolov8n_modelopt_hp_fp16_trt10.engine \
  --output-dir outputs/04_modelopt_qdq/trt10/evaluation

python3 tools/create_reference_bundle.py \
  --report outputs/04_modelopt_qdq/trt10/evaluation/precision_evaluation.json \
  --onnx ../05_torch_to_onnx/outputs/yolov8n.onnx \
  --output outputs/04_modelopt_qdq/trt10/evaluation/reference_bundle.json

cd /workspace/Learn-TensorRT
python3 12_yolov8_int8_quantization_engineering/modelopt/benchmark_trt10_evidence.py
```

The TRT10 full evaluation is mandatory because the runtime and both TensorRT reference engines
changed. Subsequent TRT10 Q/DQ candidates may reuse that new TRT10 reference bundle.
