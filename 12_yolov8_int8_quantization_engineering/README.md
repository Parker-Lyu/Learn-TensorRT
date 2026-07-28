# 12 - YOLOv8 INT8 Quantization Engineering

This lesson uses `nvcr.io/nvidia/pytorch:25.11-py3` with TensorRT 10.14.1.48 and CUDA 13.0 to
produce a reproducible YOLOv8 deployment decision. The primary workflow is post-training
quantization with explicit ONNX `QuantizeLinear`/`DequantizeLinear` (Q/DQ) nodes. Every precision
conclusion uses the same data, preprocessing contract, postprocessing, evaluator, and predeclared
quality thresholds.

## Learning Goals

1. Download and identify the COCO calibration and validation data with immutable manifests and
   SHA-256 hashes.
2. Evaluate PyTorch FP32, PyTorch FP16, TensorRT FP32, and TensorRT FP16 references on the complete
   validation split.
3. Export a Q/DQ graph with ModelOpt and build an INT8 engine as a TensorRT 10.14 strongly typed
   network.
4. Accept or reject INT8 with predeclared mAP50-95, mAP50, precision, and recall gates.
5. Benchmark only candidates that pass the quality gates, using matched runtime and measurement
   settings.

## Execution Order

Run all GPU commands from the repository root inside the course baseline container. If the
development container does not exist yet, build and enter it with NVIDIA Container Toolkit:

```bash
docker build -f docker/Dockerfile.dev -t learn-tensorrt:dev .
docker run --gpus all --rm -it --name learn-tensorrt \
  -v "$PWD:/workspace/Learn-TensorRT" -w /workspace/Learn-TensorRT \
  learn-tensorrt:dev
```

The complete, ordered procedure is in
[`docs/reproduction.md`](docs/reproduction.md). Prepare and qualify the data first:

```bash
python3 assets/coco/prepare_coco.py
python3 12_yolov8_int8_quantization_engineering/tools/prepare_calibration_dataset.py --materialize
python3 12_yolov8_int8_quantization_engineering/tools/analyze_calibration_representativeness.py
python3 12_yolov8_int8_quantization_engineering/tools/verify_preprocessing_parity.py
```

Export the static course ONNX model, establish the four references, then export, build, evaluate,
and inspect the Q/DQ INT8 candidate:

```bash
(cd 05_torch_to_onnx && python3 export_yolov8_onnx.py)
python3 12_yolov8_int8_quantization_engineering/modelopt/export_qdq.py \
  --high-precision fp16 --name yolov8n_qdq_fp16
python3 12_yolov8_int8_quantization_engineering/modelopt/build_engines.py
python3 12_yolov8_int8_quantization_engineering/compare_engines.py \
  --experiment-id modelopt_qdq_int8
python3 12_yolov8_int8_quantization_engineering/modelopt/inspect_precision.py
```

`compare_engines.py` produces unified JSON and Markdown results for the same validation split. INT8
must pass gates relative to both PyTorch FP32 and TensorRT FP16; a failing candidate is excluded
from the performance recommendation.

The committed [`reports/12_int8_quantization.md`](../reports/12_int8_quantization.md) is concise
evidence from one complete TensorRT 10.14 reproduction. It demonstrates how the quality gates,
Engine Inspector output, and matched performance measurements support a deployment decision.
Learners must regenerate every local artifact on their own GPU, driver, and container environment.

## Quality Contract

`configs/quality_contract.json` fixes the input shape, postprocessing behavior, metric
implementation, and thresholds. Do not change thresholds after seeing the result. Any change to a
manifest, model, preprocessing contract, evaluator, runtime identity, or engine requires rebuilding
and reevaluating every affected artifact.

## Outputs And Tests

Engines, timing caches, predictions, performance samples, and intermediate reports are generated
under the ignored `outputs/` directory. Run the CPU-only unit tests with:

```bash
PYTHONPATH=12_yolov8_int8_quantization_engineering \
python3 -m unittest discover -s 12_yolov8_int8_quantization_engineering/tests -v
PYTHONPATH=12_yolov8_int8_quantization_engineering \
python3 -m unittest discover -s 12_yolov8_int8_quantization_engineering/modelopt -p 'test_*.py' -v
```

TensorRT, CUDA, PyTorch, and ModelOpt versions are recorded by `configs/environments.json`. Without
the pinned container and an accessible GPU, only static checks and runtime-independent tests can be
completed; do not claim engine validation from those checks.
