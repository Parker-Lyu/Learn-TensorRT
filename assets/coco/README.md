# COCO 2017 Dataset Preparation

This directory provides the shared COCO 2017 dataset foundation used by lesson 12 and later
reports. The preparer reads its committed v1 manifest, downloads the exact 1,000-image calibration
baseline and complete validation split, converts the validation boxes to YOLO format, and verifies
every declared hash. Lesson 12 adds a separately versioned 5,000-candidate/3,000-calibration
contract on top of these shared annotations and validation assets.

Run from the repository root with Python 3.9 or newer:

```bash
python3 assets/coco/prepare_coco.py
```

The committed `data/dataset_manifest.json` is the single source of truth. Without arguments, the
script reads its exact 1,000 calibration image records and 5,000 validation image/label records,
downloads those files, recreates labels from the pinned COCO annotations, and verifies every hash.
It never resamples or rebuilds the manifest. Only download concurrency is configurable:

```bash
python3 assets/coco/prepare_coco.py \
  --workers 16
```

`data/dataset_manifest.json` is committed reproducibility metadata even though the downloaded data
around it is ignored. A missing, altered, or mismatched file stops preparation rather than silently
changing the course dataset contract.

Interrupted downloads use `.part` files and resume when the server supports byte ranges. Existing
archives and images are reused. COCO's official download endpoint is HTTP; the script pins and
verifies the SHA-256 digest of both source archives before extraction, and ZIP CRC checks provide an
additional corruption check. Downloaded files under `data/` are ignored by Git; the manifest is the
intentional exception:

```text
assets/coco/data/
  annotations/
    instances_val2017.json
  calibration/images/train2017_stratified_v1_seed42_n1000/
  validation/images/val2017/
  validation/labels/val2017/
  downloads/
  dataset_manifest.json
  preparation_summary.json
```

Allow roughly 1 GB for val2017, 150-250 MB for the default calibration images, and additional space
for the two source archives and extracted JSON annotations. Reserve at least 4 GB while preparing
the default dataset. Keeping the archives makes reruns and recovery cheaper; they may be deleted
after a successful run when disk space matters, but a later clean rebuild will download them again.

The converter maps COCO's non-contiguous category IDs to YOLO class IDs 0-79, clips boxes to image
bounds, and excludes `iscrowd` regions because lesson 12's compact evaluator has no ignored-region
semantics. This makes the labels suitable for the course accuracy gate, but the evaluator is not a
drop-in replacement for the official `pycocotools` metric implementation.

After preparing these shared prerequisites, reproduce lesson 12's fixed calibration selection:

```bash
python3 12_yolov8_int8_quantization_engineering/tools/prepare_calibration_dataset.py \
  --download-candidates \
  --materialize
```

Lesson 12 then consumes its own committed manifest by default:

```bash
python3 build_int8_engine.py --enable-fp16

python3 compare_engines.py
```

Calibration images do not need labels. Do not put val2017 images into the calibration directory or
change the fixed validation split after inspecting INT8 results. COCO is representative for this
course's official YOLOv8n model; a deployment-specific model should use images from its target
cameras and operating conditions.

Run the CPU-only preparer tests without downloading COCO:

```bash
python3 -m unittest discover -s assets/coco/tests -v
```
