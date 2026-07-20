# 12 - YOLOv8 INT8 Calibration and Accuracy Recovery

Goal: build reproducible legacy-calibrated and explicit Q/DQ INT8 TensorRT engines, measure their
quality against a fixed labeled validation split, and recover unacceptable PTQ accuracy loss with
controlled calibration-data, ModelOpt, and mixed-precision experiments.

The course path uses a reproducibly selected COCO train2017 calibration subset and the complete,
human-labeled COCO val2017 split. The accuracy gate is based only on annotations independent of the
evaluated model.

## Artifacts

- `assets/coco/prepare_coco.py`: reads the committed canonical manifest, downloads its exact COCO
  images, converts val2017 boxes to YOLO format, and verifies every declared hash.
- `dataset_manifest.py`: creates and validates a manifest for real calibration images and YOLO-format
  validation labels; byte-identical split overlap is rejected.
- `build_int8_engine.py`: implements entropy and MinMax calibrators, algorithm-aware calibration
  cache identity, named FP16 precision profiles, strict precision constraints, persistent TensorRT
  timing cache, engine metadata, and detailed Engine Inspector evidence.
- `compare_engines.py`: runs PyTorch plus TensorRT FP32, FP16, and INT8 over the complete manifest,
  or validates and reuses an identity-linked reference report when only a new INT8 candidate needs
  evaluation. Every backend uses the same letterbox, decode, confidence, NMS IoU, and
  maximum-detection settings.
- `evaluation.py`: compact NumPy prediction storage plus reusable YOLO-label parsing, IoU matching,
  101-point AP, mAP50-95, mAP50, precision, recall, and tensor-drift calculations.
- `precision_recovery/`: ordered experiment implementations and the reproducible recovery log.
- `tests/`: focused manifest, evaluator, cache-identity, reference-reuse, and precision-profile
  tests.

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

Entropy remains the default for the canonical baseline. Select MinMax explicitly for a controlled
candidate:

```bash
python3 build_int8_engine.py \
  --calibrator minmax \
  --manifest outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json \
  --cache outputs/minmax.cache \
  --output outputs/minmax.engine \
  --enable-fp16
```

Both commands default to the shared COCO manifest. The evaluator writes
`outputs/precision_evaluation.json` as the machine-readable source of truth and
`outputs/precision_evaluation.md` as a concise table. A regression gate failure still writes both
reports and exits with status `2`.

The complete evaluation runs 5,000 images through four backends and prints progress every 100
images. Predictions use fixed-capacity structured NumPy buffers instead of millions of Python
objects; metric matching groups and sorts each class once before evaluating the IoU thresholds.

During recovery, unchanged PyTorch, FP32, and FP16 results may be reused from a full report. The
evaluator verifies the manifest, artifact hashes, software versions, input shape, evaluation
settings, metric implementation, and thresholds before it runs only the new candidate:

```bash
python3 compare_engines.py \
  --reference-report outputs/reference/precision_evaluation.json \
  --int8-engine outputs/candidate.engine \
  --output-dir outputs/candidate_evaluation
```

Candidate-only mode preserves the task-level gate but intentionally omits newly computed FP32 raw
tensor drift and changed-example diagnostics. Run the full four-backend command whenever those
diagnostics or any reference identity changes.

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

## Recorded Accuracy Recovery

The canonical Entropy INT8 engine failed the predeclared accuracy gate while FP16 passed. Recovery
experiments kept the ONNX model, complete val2017 split, postprocessing settings, and gate fixed.
Detailed commands, artifact identities, coverage statistics, Inspector evidence, and reports are in
[`precision_recovery/README.md`](precision_recovery/README.md).

| Experiment | mAP50-95 | mAP50 | Recall | Gate |
| --- | ---: | ---: | ---: | --- |
| Entropy, canonical 1,000 images | 0.3179 | 0.4560 | 0.7858 | FAIL |
| Entropy, coverage-aware 3,000 images | 0.3247 | 0.4651 | 0.7940 | FAIL |
| MinMax, coverage-aware 3,000 images | 0.3444 | 0.4892 | 0.7936 | FAIL |
| MinMax, final box outputs FP16 | 0.3452 | 0.4893 | 0.7936 | FAIL |
| MinMax, final box and class outputs FP16 | 0.3447 | 0.4892 | 0.7936 | FAIL |
| MinMax, complete detection head FP16 | 0.3463 | 0.4897 | 0.7946 | FAIL |
| ModelOpt max Q/DQ, TRT8.6 INT8+FP16 | 0.3453 | 0.4931 | 0.7998 | PASS |
| ModelOpt native FP16 Q/DQ, TRT10 INT8+FP16 | 0.3452 | 0.4937 | 0.8011 | PASS |

The final mixed-precision candidate passed mAP50-95, precision, and recall, but its mAP50 drop was
`0.02057` against the allowed `0.02`. The threshold was not loosened after observing the result.
Matched `trtexec` measurements showed approximately `9.9%` lower mean latency and `20.7%` higher
throughput than FP16, but performance cannot override the failed quality gate.

The legacy-calibrator sequence therefore retained FP16 as its release decision. The subsequent
ModelOpt max-calibrated explicit Q/DQ candidate produced a new ONNX identity, was built as a
TensorRT 8.6 INT8+FP16 engine, and passed every unchanged accuracy threshold on all 5,000 validation
images. It was the first INT8 candidate eligible on quality.

The matched comparison used a 500 ms warmup and 120 measured `trtexec` iterations. The Q/DQ engine
reached `618.236 qps`, compared with `650.348 qps` for FP16, and its mean GPU compute time was
`1.602 ms` versus `1.523 ms`. The passing Q/DQ candidate is therefore retained as accuracy-recovery
evidence. A follow-up native FP16-high-precision Q/DQ export was then evaluated with new matched
TensorRT 10.14 references. It also passed the complete gate: its deltas versus PyTorch were
`-0.01789` mAP50-95, `-0.01648` mAP50, `+0.00128` precision, and `-0.00859` recall.

The TensorRT 10 matched comparison reached `635.628 qps` for FP16 and `507.842 qps` for native Q/DQ
INT8+FP16. Mean GPU compute was `1.559 ms` and `1.951 ms`, respectively. Inspector evidence showed
that native FP16 high-precision tensors reduced FP32-output compute, but the candidate still had 87
reformats, including 41 with Q/DQ origin. The final deployment decision therefore remains FP16:
both explicit-Q/DQ INT8 candidates are quality-eligible evidence, but neither provides a matched
performance benefit.

This sequence is intentional engineering evidence rather than a collection of ad hoc builds: the
lesson demonstrates reproducible INT8 calibration, identity-linked evaluation, a defensible FP16
fallback when legacy PTQ failed, and explicit Q/DQ recovery without changing the quality contract.

## Acceptance Criteria

- The INT8 engine and cache are reproducibly generated from a hashed calibration split.
- Calibration and validation manifests are saved, versioned, labeled where required, and have no
  image-hash overlap.
- PyTorch, TensorRT FP32, FP16, and INT8 run on the same complete fixed validation split with
  identical postprocessing settings.
- JSON records mAP50-95, mAP50, precision, recall, backend deltas, latency, drift, and inspection
  examples.
- Predeclared thresholds determine the process exit status.
- Calibration algorithm, precision constraints, engine identity, and reusable reference identity
  are recorded and validated.
- Accuracy loss and recovery are explained through the legacy FP16 fallback decision, the passing
  explicit-Q/DQ candidates, and the final matched evidence that retains FP16 for deployment.
