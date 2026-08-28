# 32 - Final Portfolio Case Study

## Purpose

This lesson assembles the validated YOLOv8n deployment into a reproducible portfolio report and an
optional multi-stage delivery image. It consumes generated evidence from lessons 05, 12, 15,
21, 17, and 29; it never treats an absent engine or failed GPU check as a successful deployment.

## Prerequisites

- Complete the checkpoint reports and only the elective evidence that will be presented. Follow
  each checkpoint README to collect its required evidence, then generate the three required inputs:

  ```bash
  python3 12_end_to_end_validation_report/generate_report.py
  python3 15_precision_performance_report/generate_report.py
  python3 22_pipeline_performance_report/generate_report.py
  ```

- Build the required lesson 11, 17, and 29 artifacts in the pinned development environment.
- Lesson 31 profiling reports are optional portfolio evidence; its CUDA correctness and tooling
  tests remain part of the local verification matrix.

## Deliverables

- `generate_case_study.py` evidence-driven report generator
- Local verification tools and focused report tests
- Multi-stage runtime `Dockerfile` and engine-delivery helper
- `reports/32_final_portfolio_case_study.md` generated case study

## Generate the Report

Run these commands from the repository root **inside the pinned `learn-tensorrt:25.11`
development environment**. The existing development container is sufficient; this lesson does not
require rebuilding that image.

First create the static FP16 engine used by the lesson 11 C++ runner:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py
python3 05_torch_to_onnx/validate_onnx_runtime.py
python3 06_trtexec_engine/prepare_fp16_onnx.py --models static
./32_final_portfolio_case_study/build_delivery_engine.sh
```

Build the local checkpoint matrix (generated `build/` directories are ignored):

```bash
./32_final_portfolio_case_study/build_local_checks.sh
```

For the real C ABI inference check in Lesson 29, also build the dynamic engine expected by that
lesson:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py --dynamic
python3 05_torch_to_onnx/validate_onnx_runtime.py
./17_dynamic_batching/build_dynamic_engine.sh
```

Then collect the matrix and render the report:

```bash
python3 32_final_portfolio_case_study/run_local_checks.py
python3 32_final_portfolio_case_study/generate_case_study.py
```

The report is written to `reports/32_final_portfolio_case_study.md`; raw local-check output is
written to `32_final_portfolio_case_study/outputs/local_checks.json`. The JSON records each command,
return code, duration, GPU/driver/compute capability, CUDA Toolkit, TensorRT, container identity,
and host CPU metadata. A missing GPU or engine remains a failed check and is not silently replaced
with a CPU-only claim.

### Delivery image

The optional image uses `nvcr.io/nvidia/pytorch:25.11-py3` as the reproducible TensorRT 10.14/CUDA
13.0 builder and `nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04` as the CUDA runtime base. It copies
only the lesson 11 executable, TensorRT 10 runtime library, OpenCV runtime packages, the generated
FP16 engine, and `assets/img.jpeg` into the final stage. The delivery helper builds directly from
the validated lesson 06 static AutoCast ONNX model with `trtexec --stronglyTyped`. Generate that
model in lesson 06 before rebuilding the delivery engine; the deprecated weakly typed `--fp16`
route remains available there for compatibility testing. The helper rejects an ONNX file unless
the matching lesson 06 validation report passed and its SHA256 matches the model.

After the static engine exists, build and record image/platform identity:

```bash
DEVELOPMENT_IMAGE=learn-tensorrt:25.11 \
  ./32_final_portfolio_case_study/measure_images.sh
```

The default `DEVELOPMENT_IMAGE` in the Dockerfile is the pinned upstream image, so a third party
can reproduce the build without a locally named image. The script writes image sizes and a platform
manifest under `32_final_portfolio_case_study/outputs/`. Run the image on an NVIDIA host:

```bash
docker run --rm --gpus all -v "$PWD/32_final_portfolio_case_study/outputs:/outputs" \
  learn-tensorrt-runtime:10.14
```

The serialized engine is GPU- and TensorRT-build specific. Rebuild it in the target environment
when the deployment GPU, driver, CUDA, or TensorRT runtime changes.

## Outputs

- `outputs/local_checks.json`, platform manifests, and image-size evidence are ignored generated artifacts.
- `reports/32_final_portfolio_case_study.md` is ignored and regenerated from the current evidence.

## Tests

Run the Python tests from the repository root:

```bash
python3 -m unittest discover -s 32_final_portfolio_case_study/tests -v
```

The unit tests validate report inputs, precision-decision logic, local-check coverage, and the
multi-stage Dockerfile without requiring ignored reports to exist in a clean clone.

## Checkpoints

- Trace each report number to its generated raw artifact and recorded platform identity.
- Explain why static engine packaging is separate from the dynamic batching and C ABI lessons.
- Inspect the final image and identify which development tools and source files were intentionally
  excluded.
- Repeat the case study on a second GPU and explain which measurements must be regenerated.
