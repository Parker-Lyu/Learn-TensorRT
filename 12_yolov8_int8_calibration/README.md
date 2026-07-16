# 12 - YOLOv8 INT8 Calibration and Accuracy Gate

Goal: build an entropy-calibrated INT8 TensorRT engine, then decide whether it is releasable using
latency, tensor drift, and task-level accuracy on one fixed labeled validation split.

The course path uses a reproducibly selected COCO train2017 calibration subset and the complete,
human-labeled COCO val2017 split. The accuracy gate is based only on annotations independent of the
evaluated model.

## Artifacts

- `assets/coco/prepare_coco.py`: reads the committed canonical manifest, downloads its exact COCO
  images, converts val2017 boxes to YOLO format, and verifies every declared hash.
- `dataset_manifest.py`: creates and validates a manifest for real calibration images and YOLO-format
  validation labels; byte-identical split overlap is rejected.
- `build_int8_engine.py`: implements `IInt8EntropyCalibrator2`, builds INT8 with optional FP16
  fallback, and rejects stale calibration caches using a model/data/shape/preprocessing cache key.
- `compare_engines.py`: runs PyTorch plus TensorRT FP32, FP16, and INT8 over the complete manifest,
  using the same letterbox, decode, confidence, NMS IoU, and maximum-detection settings.
- `evaluation.py`: compact NumPy prediction storage plus reusable YOLO-label parsing, IoU matching,
  101-point AP, mAP50-95, mAP50, precision, recall, and tensor-drift calculations.
- `tests/test_evaluation.py`: focused metric and invalid-label tests.

Downloaded datasets stay in ignored `assets/coco/data/`; engines, calibration tables, and raw
evaluation results stay in this lesson's ignored `outputs/` directory.

## Prerequisites

Use the pinned TensorRT development container from lesson 00. First create the model artifacts:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py
python3 06_trtexec_engine/build_and_benchmark.py --builds static_fp32 static_fp16
```

## Prepare The Fixed COCO Dataset

Run once from the repository root. Existing complete downloads are reused:

```bash
python3 assets/coco/prepare_coco.py
```

The default dataset contains 1,000 deterministic train2017 calibration images with all 80 COCO
categories represented and all 5,000 val2017 images with converted human annotations. Its hashed
manifest is `assets/coco/data/dataset_manifest.json`; see `assets/coco/README.md` for layout,
integrity checks, disk use, and command options.

## Build And Evaluate

Run from this lesson directory after preparing the model artifacts and COCO data:

```bash
python3 build_int8_engine.py --enable-fp16
python3 compare_engines.py
```

Both commands default to the shared COCO manifest. The evaluator writes
`outputs/precision_evaluation.json` as the machine-readable source of truth and
`outputs/precision_evaluation.md` as a concise table. A regression gate failure still writes both
reports and exits with status `2`.

The complete evaluation runs 5,000 images through four backends and prints progress every 100
images. Predictions use fixed-capacity structured NumPy buffers instead of millions of Python
objects; metric matching groups and sorts each class once before evaluating the IoU thresholds.

Run the CPU-only focused tests without a GPU engine:

```bash
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

## Dataset Design

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
  dataset, include enough labeled examples for every important class and hard case; a few sample
  images are not sufficient to support a release decision.

These counts are practical starting points, not universal thresholds. Distribution coverage matters
more than collecting many near-duplicate frames. Sample across different videos, time periods,
cameras, and operating conditions, and avoid placing adjacent or derived frames from the same source
scene into both calibration and validation. Keep the validation split unchanged between FP32, FP16,
and INT8 comparisons so metric deltas remain comparable.

For a custom deployment-domain dataset, create its saved manifest separately (nested image/label
paths are supported):

```bash
python3 dataset_manifest.py \
  --calibration-dir /datasets/project/calibration/images \
  --validation-dir /datasets/coco/val2017 \
  --labels-dir /datasets/coco/val2017-yolo-labels \
  --dataset-id coco-val2017-fixed-v1 \
  --output data/dataset_manifest.json
```

Override the default paths or predeclared gates when an experiment requires it:

```bash
python3 build_int8_engine.py \
  --manifest ../assets/coco/data/dataset_manifest.json \
  --output outputs/yolov8n_static_int8.engine \
  --cache outputs/yolov8n_int8_calibration.cache \
  --enable-fp16

python3 compare_engines.py \
  --manifest ../assets/coco/data/dataset_manifest.json \
  --max-map50-95-drop 0.02 \
  --max-map50-drop 0.02 \
  --max-precision-drop 0.03 \
  --max-recall-drop 0.03
```

Declare these thresholds before inspecting the final comparison. Defaults are teaching examples,
not universal production limits. Evaluation defaults (`confidence=0.001`, `NMS IoU=0.7`, and
`max_detections=300`) are recorded in JSON and applied identically to every backend.

The reported AP is a documented course COCO-like 101-point metric. It excludes `iscrowd` regions
and official COCO area-range/ignore semantics, so it must not be presented as an official
`pycocotools` COCO score. Its purpose is a fixed, identical regression gate across precisions.

## Reading the Evidence

For every backend, the report contains absolute mAP50-95, mAP50, precision, recall, deltas from the
PyTorch reference, mean/P50/P90 latency, and pass/fail. TensorRT results also contain FP32-relative
raw tensor drift. Images with changed detection counts/classes or high P99 drift are listed for
visual inspection.

Latency includes each runtime wrapper's H2D transfer, inference, D2H transfer, synchronization, and
wrapper overhead, but excludes image loading, preprocessing, and decoding. It is diagnostic rather
than the authoritative performance comparison; 12a uses matched `trtexec` sampling for FP32, FP16,
and INT8.

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
