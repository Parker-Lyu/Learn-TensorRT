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
development environment**. Each command includes its purpose and a capture from the local run.

Export the static YOLOv8n ONNX model:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py
```

Example output:
```text
Export complete (1.0s)
Results saved to /workspace/Learn-TensorRT/assets
dynamic: False
```

Validate the exported graph against PyTorch:
```bash
python3 05_torch_to_onnx/validate_onnx_runtime.py
```
Example output:
```text
input: (1, 3, 640, 640) float32
onnxruntime: (1, 84, 8400) float32
max abs error: 0.00155640
allclose(rtol=0.001, atol=0.001): True
report: /workspace/Learn-TensorRT/05_torch_to_onnx/outputs/validation_report.json
```

Prepare the static strongly typed FP16 ONNX model:
```bash
python3 06_trtexec_engine/prepare_fp16_onnx.py --models static
```
Example output:
```text
static: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/yolov8n_static_autocast_fp16.onnx
validation: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/static_fp16_onnx_validation.json
```

Build the delivery engine used by the Lesson 11 runner:
```bash
./32_final_portfolio_case_study/build_delivery_engine.sh
```
```text
[I] TensorRT version: 10.14.1
[I] Engine generation completed in 1.32574 seconds.
[I] Created engine with size: 12.6623 MiB
&&&& PASSED TensorRT.trtexec [TensorRT v101401]
```

Build all native artifacts used by the local checkpoint matrix (build directories are ignored):

```bash
./32_final_portfolio_case_study/build_local_checks.sh
```
```text
-- Build files have been written to: /workspace/Learn-TensorRT/29_cpp_shared_library_python_binding/build
[1/4] Building CXX object CMakeFiles/trt_inference.dir/.../batch_layout.cpp.o
[2/4] Building CXX object CMakeFiles/trt_inference.dir/src/trt_c_api.cpp.o
[3/4] Building CXX object CMakeFiles/trt_inference.dir/.../dynamic_batch_runner.cpp.o
[4/4] Linking CXX shared library libtrt_inference.so
```

For the Lesson 29 C ABI check, export the dynamic-batch model (the `--dynamic` parameter variant):

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py --dynamic
```

Validate the dynamic export:
```bash
python3 05_torch_to_onnx/validate_onnx_runtime.py
```
Example output:
```text
allclose(rtol=0.001, atol=0.001): True
report: /workspace/Learn-TensorRT/05_torch_to_onnx/outputs/validation_report.json
```

Build the dynamic engine expected by Lesson 29:
```bash
./17_dynamic_batching/build_dynamic_engine.sh
```
Example output:
```text
&&&& PASSED TensorRT.trtexec [TensorRT v101401]
```

Collect the local verification matrix:

```bash
python3 32_final_portfolio_case_study/run_local_checks.py
```
<details><summary>Example output</summary>

```text
{
  "passed": true,
  "checks": ["lesson11 preprocessing/postprocessing", "lesson16 concurrency", "lesson17 batching",
    "lesson18 async pipeline", "lesson19 multistream", "lesson20 CUDA preprocess",
    "lesson29 ctypes inference", "lesson31 Nsight Compute kernel analysis"]
}
```
</details>

Render the evidence-driven case-study report:
```bash
python3 32_final_portfolio_case_study/generate_case_study.py
```
Example output:
```text
wrote /workspace/Learn-TensorRT/reports/32_final_portfolio_case_study.md
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

After the static engine exists, run this command on the **host** (it invokes Docker) to build and
record image/platform identity:

```bash
DEVELOPMENT_IMAGE=learn-tensorrt:25.11 \
  ./32_final_portfolio_case_study/measure_images.sh
```
```text
Successfully built fa6cf5964fca
Successfully tagged learn-tensorrt-runtime:10.14
["learn-tensorrt-runtime:10.14"] sha256:fa6cf5964fca... 596051869
```

The default `DEVELOPMENT_IMAGE` in the Dockerfile is the pinned upstream image, so a third party
can reproduce the build without a locally named image. The script writes image sizes and a platform
manifest under `32_final_portfolio_case_study/outputs/`. Run the image on an NVIDIA host:

```bash
docker run --rm --gpus all -v "$PWD/32_final_portfolio_case_study/outputs:/outputs" \
  learn-tensorrt-runtime:10.14
```
<details><summary>Example output</summary>

```text
Engine: /app/model.engine
Input tensor: images
Output tensor: output0
Detections: 6
Last latency ms: preprocess=2.85145, h2d=0.26096, enqueue_host=29.0395, gpu_compute=28.7961, d2h=0.146336, postprocess=0.411843, total=33.4404
Output image: /outputs/input_yolov8_trt_cpp.jpg
JSON report: /outputs/detections.json
```
</details>

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
