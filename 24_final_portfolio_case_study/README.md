# 24 - Final Portfolio Case Study

This lesson assembles the validated YOLOv8n deployment into a reproducible portfolio report and an
optional multi-stage delivery image. It consumes generated evidence from lessons 05, 06, 10a, 12a,
17a, 14, and 21; it never treats an absent engine or failed GPU check as a successful deployment.

## Reproduce the evidence

Run these commands from the repository root **inside the pinned `learn-tensorrt:25.11`
development environment**. The existing development container is sufficient; this lesson does not
require rebuilding that image.

First create the static engine used by the lesson 10 C++ runner:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py
python3 06_trtexec_engine/build_and_benchmark.py --builds static_fp16
```

Build the local checkpoint matrix (generated `build/` directories are ignored):

```bash
./24_final_portfolio_case_study/build_local_checks.sh
```

For the real C ABI inference check, also build the dynamic engine expected by lesson 21:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py --dynamic
./14_dynamic_batching/build_dynamic_engine.sh
```

Then collect the matrix and render the report:

```bash
python3 24_final_portfolio_case_study/run_local_checks.py
python3 24_final_portfolio_case_study/generate_case_study.py
python3 -m unittest discover -s 24_final_portfolio_case_study/tests -v
```

The report is written to `reports/24_final_portfolio_case_study.md`; raw local-check output is
written to `24_final_portfolio_case_study/outputs/local_checks.json`. The JSON records each command,
return code, duration, GPU/driver/compute capability, CUDA Toolkit, TensorRT, container identity,
and host CPU metadata. A missing GPU or engine remains a failed check and is not silently replaced
with a CPU-only claim.

## Delivery image

The optional image uses `nvcr.io/nvidia/pytorch:25.11-py3` as the reproducible TensorRT 10.14/CUDA
13.0 builder and `nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04` as the CUDA runtime base. It copies
only the lesson 10 executable, TensorRT 10 runtime library, OpenCV runtime packages, the generated
static FP16 engine, and `assets/img.jpeg` into the final stage.

After the static engine exists, build and record image/platform identity:

```bash
DEVELOPMENT_IMAGE=learn-tensorrt:25.11 \
  ./24_final_portfolio_case_study/measure_images.sh
```

The default `DEVELOPMENT_IMAGE` in the Dockerfile is the pinned upstream image, so a third party
can reproduce the build without a locally named image. The script writes image sizes and a platform
manifest under `24_final_portfolio_case_study/outputs/`. Run the image on an NVIDIA host:

```bash
docker run --rm --gpus all -v "$PWD/24_final_portfolio_case_study/outputs:/outputs" \
  learn-tensorrt-runtime:10.14
```

The serialized engine is GPU- and TensorRT-build specific. Rebuild it in the target environment
when the deployment GPU, driver, CUDA, or TensorRT runtime changes.

## Learning checkpoints

- Trace each report number to its generated raw artifact and recorded platform identity.
- Explain why static engine packaging is separate from the dynamic batching and C ABI lessons.
- Inspect the final image and identify which development tools and source files were intentionally
  excluded.
- Repeat the case study on a second GPU and explain which measurements must be regenerated.
