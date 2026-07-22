# COCO 2017 Shared Data

This directory prepares the shared COCO annotations and complete labeled val2017 split. Lesson 12
owns calibration-image selection because calibration policy is part of that lesson's quantization
experiment rather than a repository-wide dataset default.

Run from the repository root with Python 3.9 or newer:

```bash
python3 assets/coco/prepare_coco.py
```

The script downloads and verifies `annotations_trainval2017.zip` and `val2017.zip`, extracts the
validation images, converts non-crowd boxes to YOLO labels, and verifies the committed validation
manifest. Interrupted archive downloads resume when the server supports byte ranges.

Generated data is ignored by Git. The committed manifest and generated layout are:

```text
assets/coco/data/
  annotations/instances_val2017.json
  validation/images/val2017/
  validation/labels/val2017/
  downloads/
  dataset_manifest.json
  preparation_summary.json
```

After preparing these shared prerequisites, run lesson 12's selector. It samples the calibration
set from the complete train2017 annotation population and downloads only the images required to
reproduce the fixed selection:

```bash
python3 12_yolov8_int8_quantization_engineering/tools/prepare_calibration_dataset.py \
  --materialize
```

Run the CPU tests without downloading COCO:

```bash
python3 -m unittest discover -s assets/coco/tests -v
```
