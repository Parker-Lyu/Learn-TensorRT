# 12 - YOLOv8 INT8 Calibration and Accuracy Gate

Goal: build an entropy-calibrated INT8 TensorRT engine, then decide whether it is releasable using
latency, tensor drift, and task-level accuracy on one fixed labeled validation split.

The runnable smoke path creates pseudo-labels from `assets/yolov8n.pt`. It proves that calibration,
multi-backend evaluation, reporting, and failure exit codes work; because the same model creates the
labels, its metrics are not valid accuracy evidence. A real release decision requires a versioned
public or deployment-domain dataset with human-reviewed labels.

## Artifacts

- `prepare_calibration_data.py`: generates disjoint smoke splits, pseudo-labels, and a hashed
  manifest.
- `dataset_manifest.py`: creates and validates a manifest for real calibration images and YOLO-format
  validation labels; byte-identical split overlap is rejected.
- `build_int8_engine.py`: implements `IInt8EntropyCalibrator2`, builds INT8 with optional FP16
  fallback, and rejects stale calibration caches using a model/data/shape/preprocessing cache key.
- `compare_engines.py`: runs PyTorch plus TensorRT FP32, FP16, and INT8 over the complete manifest,
  using the same letterbox, decode, confidence, NMS IoU, and maximum-detection settings.
- `evaluation.py`: reusable YOLO-label parsing, IoU matching, 101-point AP, mAP50-95, mAP50,
  precision, recall, and tensor-drift calculations.
- `tests/test_evaluation.py`: focused metric and invalid-label tests.

Generated datasets, engines, calibration tables, and reports stay in ignored `data/` and `outputs/`
directories.

## Prerequisites

Use the pinned TensorRT development container from lesson 00. First create the model artifacts:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py
python3 06_trtexec_engine/build_and_benchmark.py --builds static_fp32 static_fp16
```

## Smoke Workflow

Run from this lesson directory:

```bash
python3 prepare_calibration_data.py
python3 build_int8_engine.py --enable-fp16
python3 compare_engines.py
```

The evaluator writes `outputs/precision_evaluation.json` as the machine-readable source of truth
and `outputs/precision_evaluation.md` as a concise table. A regression gate failure still writes both
reports and exits with status `2`.

Run the CPU-only focused tests without a GPU engine:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

## Real Fixed Dataset

Keep calibration and validation separate. Calibration images should represent target cameras,
lighting, compression, resolutions, normal scenes, and hard cases. Validation must be a fixed,
labeled split such as COCO val2017 or a versioned internal dataset; it must not reuse calibration
images.

Recommended starting sizes:

- Calibration: approximately 500-2,000 representative images. Calibration images do not require
  labels. Increase the set when the deployment domain contains many cameras, environments, object
  scales, lighting conditions, or rare but important cases.
- Validation: use a statistically meaningful fixed labeled split. For a general COCO YOLOv8
  comparison, the complete COCO val2017 split contains 5,000 images. For a deployment-domain
  dataset, include enough labeled examples for every important class and hard case; a few smoke
  images are not sufficient to support a release decision.

These counts are practical starting points, not universal thresholds. Distribution coverage matters
more than collecting many near-duplicate frames. Sample across different videos, time periods,
cameras, and operating conditions, and avoid placing adjacent or derived frames from the same source
scene into both calibration and validation. Keep the validation split unchanged between FP32, FP16,
and INT8 comparisons so metric deltas remain comparable.

Create the saved manifest (nested image/label paths are supported):

```bash
python3 dataset_manifest.py \
  --calibration-dir /datasets/project/calibration/images \
  --validation-dir /datasets/coco/val2017 \
  --labels-dir /datasets/coco/val2017-yolo-labels \
  --dataset-id coco-val2017-fixed-v1 \
  --output data/dataset_manifest.json
```

Build and evaluate that exact dataset:

```bash
python3 build_int8_engine.py \
  --manifest data/dataset_manifest.json \
  --output outputs/yolov8n_static_int8.engine \
  --cache outputs/yolov8n_int8_calibration.cache \
  --enable-fp16

python3 compare_engines.py \
  --manifest data/dataset_manifest.json \
  --max-map50-95-drop 0.02 \
  --max-map50-drop 0.02 \
  --max-precision-drop 0.03 \
  --max-recall-drop 0.03
```

Declare these thresholds before inspecting the final comparison. Defaults are teaching examples,
not universal production limits. Evaluation defaults (`confidence=0.001`, `NMS IoU=0.7`, and
`max_detections=300`) are recorded in JSON and applied identically to every backend.

## Reading the Evidence

For every backend, the report contains absolute mAP50-95, mAP50, precision, recall, deltas from the
PyTorch reference, mean/P50/P90 latency, and pass/fail. TensorRT results also contain FP32-relative
raw tensor drift. Images with changed detection counts/classes or high P99 drift are listed for
visual inspection.

Latency includes backend execution and required transfers, but excludes image loading,
preprocessing, and decoding. It is useful for a controlled lesson comparison; lesson 11/12a should
add longer warmups, more samples, hardware/power-state metadata, and profiler evidence.

If INT8 violates the predeclared gate:

- verify calibration and inference preprocessing are byte-for-byte equivalent;
- improve calibration coverage and inspect the listed high-drift images;
- permit mixed INT8/FP16 tactics with `--enable-fp16`;
- constrain sensitive layers to FP16/FP32 in an advanced build workflow;
- use QAT if representative PTQ calibration still fails;
- retain FP16 when INT8 speedup does not justify the accuracy loss.

## Acceptance Criteria

- The INT8 engine and cache are reproducibly generated from a hashed calibration split.
- Calibration and validation manifests are saved, versioned, labeled where required, and have no
  image-hash overlap.
- PyTorch, TensorRT FP32, FP16, and INT8 run on the same complete fixed validation split with
  identical postprocessing settings.
- JSON records mAP50-95, mAP50, precision, recall, backend deltas, latency, drift, and inspection
  examples.
- Predeclared thresholds determine the process exit status.
- Accuracy loss can be explained with a concrete FP16/mixed-precision/QAT fallback decision.
