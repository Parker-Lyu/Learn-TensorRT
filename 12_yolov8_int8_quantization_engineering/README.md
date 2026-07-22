# 12 - YOLOv8 INT8 Quantization Engineering

This lesson treats INT8 as an evidence-driven deployment decision. It builds a reproducible
calibration dataset, proves preprocessing parity, evaluates legacy TensorRT calibration and
ModelOpt explicit Q/DQ, applies one unchanged task-quality gate, and compares only
version-matched engines.

## Learning Objectives

By the end of the lesson, the evidence should answer:

1. Does the calibration set mostly follow the intended input distribution?
2. Are small, large, crowded, unusual-aspect, dark, and bright inputs still represented?
3. Are calibration and evaluation preprocessing byte-identical?
4. Does each INT8 candidate pass the predeclared detection-quality gate?
5. If quality passes, is INT8 faster than FP16 in the same TensorRT environment?

## Canonical Data Contract

Validation uses all 5,000 labeled COCO val2017 images. Calibration uses 3,000 images selected from
the complete 118,287-image train2017 annotation population. Calibration and validation are checked
for duplicate content, and generated manifests record every selected image hash.

The calibration policy intentionally combines distribution fidelity with explicit tail coverage:

- 2,400 images (80%) are a deterministic random sample from train2017's natural image
  distribution;
- 600 images (20%) are split equally across small-object, large-object, crowded,
  extreme-aspect-ratio, dark, and bright groups;
- category coverage is a minimum constraint on the natural core, not a request for uniform class
  frequency;
- dark and bright groups are measured after the exact 640x640 letterbox, RGB conversion, and
  normalization used by calibration.

The selection configuration is committed in `configs/calibration_selection.json`. The immutable
result is `data/calibration_selection.json`; normal course runs recompute the selection and require
an exact match rather than silently replacing it.

## Prepare And Inspect The Dataset

Run inside the pinned `trt_dev` container from the repository root:

```bash
python3 assets/coco/prepare_coco.py

python3 12_yolov8_int8_quantization_engineering/tools/prepare_calibration_dataset.py \
  --materialize

python3 12_yolov8_int8_quantization_engineering/tools/analyze_calibration_representativeness.py
```

The shared preparer downloads annotations and val2017. The lesson selector reads the full
train2017 annotation population, downloads a fixed brightness-screening pool and the selected
images from the official COCO endpoint, verifies hashes, and materializes only the final 3,000
images. A full train2017 image archive is not required.

Generated evidence is written under:

```text
outputs/data_preparation/
  selection_report.json
  representativeness/
    representativeness_report.json
    representativeness_report.md
    geometry_distributions.png
```

The representativeness report separates two questions: whether frequency distributions resemble a
natural sample, and whether important support regions are covered. Coverage alone is not described
as distribution fidelity.

## Preprocessing Contract

`calibration_preprocessing.py` is the calibration-side implementation used by both dataset
selection and TensorRT engine calibration. Qualification independently compares it with the
evaluation path:

```bash
python3 12_yolov8_int8_quantization_engineering/tools/verify_preprocessing_parity.py
```

The required tensor is contiguous FP32 NCHW RGB with shape `1x3x640x640`, generated using the same
linear resize, centered padding value 114, channel conversion, and `/255` normalization.

## Fixed Quality Contract

`configs/quality_contract.json` fixes the validation dataset identity, input shape, confidence and
NMS thresholds, maximum detections, metric implementation, and maximum allowed regression before
candidate results are inspected. Every evaluation records the quality-contract and manifest hashes.

Changing the calibration selection invalidates calibration caches, INT8 engines, source-model
metadata, evaluation reports, and downstream performance conclusions. Regenerate them rather than
reusing evidence with a different manifest hash.

## Experiment Sequence

The declared matrix is `configs/experiments.json`. Each stage changes one quantization decision:

1. Qualify data identity and preprocessing, then build FP32 and FP16 references.
2. Build legacy Entropy INT8+FP16.
3. Change only the legacy calibrator to MinMax.
4. Keep MinMax and constrain the complete YOLOv8 detection head to FP16.
5. Export ModelOpt explicit-Q/DQ candidates.
6. Rebuild references when moving from TensorRT 8.6 to TensorRT 10.
7. Benchmark only candidates that pass the unchanged quality gate.

TensorRT 8.6 candidates run in `trt_dev`. ModelOpt export and TensorRT 10 evidence run in the
environment declared by `configs/environments.json`. A reference bundle is reusable only while its
model, dataset manifest, quality contract, preprocessing, postprocessing, runtime, and reference
engine identities remain unchanged.

Complete commands are in [`docs/reproduction.md`](docs/reproduction.md).

## Deployment Decision Rule

Dataset statistics qualify the calibration inputs, not the engine. A calibration set is considered
usable only when the resulting engine passes the task-quality gate. A quality-passing INT8 engine is
eligible for deployment only when it also improves matched performance enough to justify its added
complexity.

After collecting fresh evidence, generate the curated case study:

```bash
python3 12_yolov8_int8_quantization_engineering/tools/generate_case_study.py
```

The generator rejects missing or identity-mismatched evidence instead of combining unrelated runs.

## Verification

Run the lesson CPU tests inside `trt_dev`:

```bash
PYTHONPATH=12_yolov8_int8_quantization_engineering \
python3 -m unittest discover \
  -s 12_yolov8_int8_quantization_engineering/tests -v
```

Run ModelOpt and TensorRT 10 focused tests inside `learn-tensorrt-modelopt`:

```bash
PYTHONPATH=12_yolov8_int8_quantization_engineering \
python3 -m unittest discover \
  -s 12_yolov8_int8_quantization_engineering/modelopt \
  -p 'test_*.py' -v
```

Generated datasets, calibration caches, engines, predictions, and benchmark captures remain in
ignored output directories. Curated reports should be committed only after all linked evidence has
been regenerated from the canonical manifest.
