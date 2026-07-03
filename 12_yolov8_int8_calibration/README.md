# 12 - YOLOv8 INT8 Calibration

You do not need to manually collect a dataset before starting this lesson. The runnable smoke path
generates a tiny local calibration and validation set from `assets/dog.webp` so the code can be
exercised end to end.

That generated set is not representative enough for a real INT8 release. For production decisions,
replace it with real images from the target camera/domain.

Goal: build INT8 TensorRT engines and understand quantization trade-offs with speed, tensor drift,
and detection-quality evidence.

Topics:

- Calibration image set
- Separate validation image set
- PTQ
- Entropy calibration and KL-divergence intuition
- `IInt8EntropyCalibrator2`
- Calibration cache/table
- INT8 engine build
- FP32, FP16, and INT8 output comparison
- FP16 versus INT8 latency
- Tensor drift summary
- Decoded box, class, and confidence comparison
- Mixed precision fallback
- QAT as the fallback when PTQ fails

## Dataset Answer

For this course lesson:

- Use `prepare_calibration_data.py` to generate a smoke calibration set.
- Use the smoke set only to verify the workflow and code.
- Use a real representative image directory before trusting INT8 accuracy.

For a real deployment, collect calibration images that match the deployment distribution:

- same camera types, viewpoints, lighting, weather, compression, and resolution patterns
- normal scenes plus hard cases
- enough class/object diversity to exercise activation ranges
- no overlap requirement with validation, but keep validation separate so calibration does not grade
  itself

## Runnable Artifacts

- `prepare_calibration_data.py`: creates `data/calibration_smoke/` and `data/validation_smoke/`.
- `build_int8_engine.py`: builds an INT8 TensorRT engine with `IInt8EntropyCalibrator2`.
- `compare_engines.py`: compares FP32, FP16, and INT8 engines on validation images.

Generated data and reports go to `data/` and `outputs/`; both are ignored by git.

## Prerequisites

Complete lessons 05, 06, and 09 first:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py
python3 06_trtexec_engine/build_and_benchmark.py --builds static_fp32 static_fp16
python3 09_yolov8_trt_python/infer_yolov8_trt.py
```

## Run

Generate smoke calibration and validation images:

```bash
python3 prepare_calibration_data.py
```

Build an INT8 engine:

```bash
python3 build_int8_engine.py
```

First-time INT8 builds can take several minutes because TensorRT must run calibration and tactic
selection. The calibration cache is saved under `outputs/` so later rebuilds can skip calibration
when the model, input shape, preprocessing, and calibration images are unchanged.

Compare FP32, FP16, and INT8:

```bash
python3 compare_engines.py
```

Use a real calibration directory when available:

```bash
python3 build_int8_engine.py \
  --calibration-dir /path/to/representative/images \
  --output outputs/yolov8n_realdata_int8.engine \
  --cache outputs/yolov8n_realdata_int8.cache
```

Use a separate validation directory:

```bash
python3 compare_engines.py \
  --validation-dir /path/to/validation/images \
  --int8-engine outputs/yolov8n_realdata_int8.engine
```

## Outputs

- `outputs/yolov8n_static_int8.engine`: generated INT8 engine.
- `outputs/yolov8n_int8_calibration.cache`: calibration cache.
- `outputs/int8_comparison_report.json`: machine-readable drift and detection comparison.
- `outputs/int8_comparison_report.md`: human-readable summary.

## How To Interpret Results

Tensor drift alone is not a release decision. Look at:

- whether top class changes on important images
- whether detection count drops unexpectedly
- whether confidence drops near threshold
- whether box coordinates move enough to affect downstream logic
- whether INT8 actually improves latency on the target GPU

If INT8 causes a severe recall drop:

- verify calibration preprocessing exactly matches inference preprocessing
- increase calibration image diversity
- inspect high-drift examples visually
- allow FP16 fallback while building INT8
- try sensitive-layer fallback or QAT in a later advanced lesson
- keep FP16 if INT8 speedup is small or accuracy cost is too high

## Checkpoints

- Open `data/manifest.json` and explain why the smoke set is not representative.
- Delete `outputs/yolov8n_int8_calibration.cache` and rebuild to see calibration run again.
- Compare FP16 and INT8 detection outputs on the smoke validation set.
- Replace `--calibration-dir` with real images and compare the report.

Acceptance criteria:

- An INT8 engine is generated.
- A representative calibration-set requirement is documented.
- A validation image set is documented separately from the calibration set.
- A short report compares FP32, FP16, and INT8 tensor drift and detection quality.
- The report lists changed-detection examples that deserve visual inspection.
- You can explain what to do when INT8 causes a severe recall drop.
