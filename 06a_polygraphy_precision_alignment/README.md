# 06a - Polygraphy Precision Alignment

This lesson uses Polygraphy to compare ONNX Runtime and TensorRT outputs with one exact same
preprocessed YOLOv8n input tensor.

Goal: create a repeatable accuracy-debug workflow for deciding whether backend differences are
acceptable numerical drift or a deployment bug.

Scope: this is a single-input tensor alignment lesson. It proves that one controlled ONNX Runtime
run and one controlled TensorRT run are using comparable artifacts and producing numerically similar
raw model outputs. It is not a dataset-level accuracy regression test and it does not replace
detection-quality validation on many images.

Topics:

- Polygraphy model inspection
- ONNX Runtime versus TensorRT comparison
- Saving input and output tensors
- FP32 and FP16 single-input drift analysis
- Absolute and relative tolerance selection
- First-mismatch debugging workflow
- Reproducible command logs for benchmark reports

## Why This Matters

TensorRT deployment is not finished when an engine builds. The engine must still produce outputs
that match the validated ONNX model closely enough for the target task.

This lesson keeps the comparison narrow and honest:

```text
lesson 05 preprocessed tensor
  -> Polygraphy data loader reads .npy
  -> ONNX Runtime output
  -> TensorRT output
  -> error summary and precision note
```

The raw YOLO output is compared before decode, NMS, visualization, or coordinate mapping. That makes
it easier to tell whether drift starts in model execution or in later postprocessing code.

In production work, this single-sample check is only the first gate:

```text
single tensor alignment
  -> multi-image numerical drift statistics
  -> decoded box/class/score comparison
  -> dataset-level detection quality report
```

Lesson 12 extends this idea when comparing FP32, FP16, and INT8 engines. Lesson 24 should include
both this precision-alignment note and later accuracy-regression evidence.

## Directory Layout

- `load_npy_input.py`: Polygraphy data loader that feeds the lesson 05 NCHW `.npy` tensor.
- `align_precision.py`: runs Polygraphy inspection and inference commands, saves logs, and writes a
  compact precision report.
- `polygraphy_cli_compat.py`: local launcher that keeps Polygraphy working with the repository's
  NumPy 2.x environment without changing system packages.
- `outputs/`: generated runner outputs, logs, JSON reports, and Markdown notes. This folder is
  ignored by git.
- `../05_torch_to_onnx/outputs/yolov8n.onnx`: validated ONNX model from lesson 05.
- `../06_trtexec_engine/outputs/yolov8n_static_fp32.engine`: default serialized TensorRT engine
  from lesson 06.

## Prerequisites

Run the lesson 05 export and validation first:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py
python3 05_torch_to_onnx/validate_onnx_runtime.py
```

Build at least one TensorRT engine with lesson 06:

```bash
python3 06_trtexec_engine/build_and_benchmark.py --builds static_fp32
```

Polygraphy, ONNX Runtime, TensorRT Python, and `trtexec` should come from the pinned TensorRT
development container used in lesson 00.

## Input Tensor

This lesson directly reuses the controlled input tensor saved by lesson 05:

```text
05_torch_to_onnx/outputs/input_nchw_float32.npy
```

`align_precision.py` passes this `.npy` file to Polygraphy through `load_npy_input.py` and
`--data-loader-script`. The tensor remains in NumPy's binary format; no intermediate input JSON is
generated.

Use a different input tensor when experimenting with another preprocessed sample:

```bash
python3 06a_polygraphy_precision_alignment/align_precision.py \
  --input-npy path/to/input_nchw_float32.npy \
  --skip-trt
```

Override the input tensor name if the ONNX inspection report shows a different name:

```bash
python3 06a_polygraphy_precision_alignment/align_precision.py --input-name images --skip-trt
```

## Smoke Test ONNX Runtime

Run only the ONNX Runtime side when you want to verify Polygraphy setup before using TensorRT:

```bash
python3 06a_polygraphy_precision_alignment/align_precision.py --skip-trt
```

This writes:

- `outputs/inspect_onnx.log`
- `outputs/run_onnxrt.log`
- `outputs/onnxrt_outputs.json`
- `outputs/precision_report.json`
- `outputs/precision_alignment_note.md`

## Compare ONNX Runtime And TensorRT

Compare the lesson 05 ONNX model against the lesson 06 FP32 engine:

```bash
python3 06a_polygraphy_precision_alignment/align_precision.py
```

The default comparison uses:

```text
ONNX:   05_torch_to_onnx/outputs/yolov8n.onnx
Engine: 06_trtexec_engine/outputs/yolov8n_static_fp32.engine
Input:  05_torch_to_onnx/outputs/input_nchw_float32.npy
```

The command writes:

- `outputs/inspect_onnx.log`
- `outputs/inspect_engine.log`
- `outputs/run_onnxrt.log`
- `outputs/compare_onnxrt_trt.log`
- `outputs/onnxrt_outputs.json`
- `outputs/trt_compare_outputs.json`
- `outputs/precision_report.json`
- `outputs/precision_alignment_note.md`

Use a different engine, for example the FP16 engine from lesson 06:

```bash
python3 06a_polygraphy_precision_alignment/align_precision.py \
  --engine 06_trtexec_engine/outputs/yolov8n_static_fp16.engine \
  --rtol 1e-2 \
  --atol 1e-2 \
  --keep-going
```

`--keep-going` is useful for FP16 or INT8 experiments because Polygraphy may return a nonzero status
when tolerance fails, but the mismatch evidence is still valuable.

Let Polygraphy build a temporary TensorRT engine from ONNX when a serialized engine is not available:

```bash
python3 06a_polygraphy_precision_alignment/align_precision.py --trt-mode build
```

The serialized lesson 06 engine is preferred for normal course work because it compares the exact
artifact that later C++ lessons will load.

## Tolerance Notes

Start strict for FP32:

```bash
python3 06a_polygraphy_precision_alignment/align_precision.py --rtol 1e-3 --atol 1e-3
```

For FP16, expect larger drift:

```bash
python3 06a_polygraphy_precision_alignment/align_precision.py \
  --engine 06_trtexec_engine/outputs/yolov8n_static_fp16.engine \
  --rtol 1e-2 \
  --atol 1e-2 \
  --keep-going
```

Do not loosen tolerance just to make a command pass. The default Polygraphy comparison uses
elementwise relative and absolute tolerance, while `precision_report.json` still records max error,
mean error, P99 error, close fraction, and the index of the largest mismatch. This report can show
whether the backend execution is suspicious, but it cannot prove final detector quality by itself.
Decide whether decoded detections remain acceptable on a representative image set before calling
FP16 or INT8 drift acceptable.

## Expected Report Fields

`precision_report.json` records:

- ONNX path
- engine path
- input `.npy` path and input tensor name
- TensorRT mode
- command lines and log paths
- runner output artifacts
- output names and shapes
- max, mean, median, and P99 absolute error
- largest-mismatch index and values
- tolerance settings
- `np.allclose` result
- likely-cause note

`precision_alignment_note.md` is the short human-readable note that should feed the final benchmark
report.

## Checkpoints

- Run `--skip-trt` and confirm Polygraphy can execute the ONNX model with the saved input tensor.
- Compare FP32 ONNX Runtime and FP32 TensorRT output, then explain the largest mismatch.
- Repeat with the FP16 engine and explain why a different tolerance may be reasonable.
- Intentionally set `--rtol 1e-8 --atol 1e-8` and inspect the generated mismatch evidence.
- Confirm the ONNX model and TensorRT engine came from the same export before debugging
  postprocessing.
- Save the final `precision_alignment_note.md` as evidence for lesson 24.
- Explain why a single-input allclose result is useful for debugging but insufficient for release
  approval.

Acceptance criteria:

- Polygraphy can run the YOLO ONNX model with ONNX Runtime.
- Polygraphy can run the same model or engine with TensorRT.
- ONNX Runtime and TensorRT outputs are compared using the same input tensor.
- Any mismatch is summarized with max error, mean error, tolerance, and likely cause.
- The final note states that this is single-input evidence and points to later multi-image
  detection-quality validation before deployment approval.

## Polygraphy Command Reference

`align_precision.py` calls Polygraphy through `polygraphy_cli_compat.py`, which applies the local
NumPy compatibility patch and then forwards arguments to Polygraphy. The displayed command in the
lesson logs starts with `polygraphy`, but the actual Python launcher is:

```bash
python3 06a_polygraphy_precision_alignment/polygraphy_cli_compat.py ...
```

The default ONNX inspection command is:

```bash
polygraphy inspect model 05_torch_to_onnx/outputs/yolov8n.onnx \
  --model-type onnx \
  --show layers \
  --log-file 06a_polygraphy_precision_alignment/outputs/inspect_onnx.log \
  --log-format no-colors
```

The ONNX Runtime smoke run uses the `.npy` input through the lesson data loader:

```bash
POLYGRAPHY_INPUT_NPY=05_torch_to_onnx/outputs/input_nchw_float32.npy \
POLYGRAPHY_INPUT_NAME=images \
polygraphy run 05_torch_to_onnx/outputs/yolov8n.onnx \
  --onnxrt \
  --data-loader-script 06a_polygraphy_precision_alignment/load_npy_input.py \
  --save-outputs 06a_polygraphy_precision_alignment/outputs/onnxrt_outputs.json \
  --log-file 06a_polygraphy_precision_alignment/outputs/run_onnxrt.log \
  --log-format no-colors
```

When `--trt-mode engine` is used, the serialized TensorRT engine is inspected first:

```bash
polygraphy inspect model 06_trtexec_engine/outputs/yolov8n_static_fp32.engine \
  --model-type engine \
  --show layers \
  --log-file 06a_polygraphy_precision_alignment/outputs/inspect_engine.log \
  --log-format no-colors
```

The default engine comparison command reuses the saved ONNX Runtime output as the reference and
feeds the same `.npy` input to TensorRT:

```bash
POLYGRAPHY_INPUT_NPY=05_torch_to_onnx/outputs/input_nchw_float32.npy \
POLYGRAPHY_INPUT_NAME=images \
polygraphy run 06_trtexec_engine/outputs/yolov8n_static_fp32.engine \
  --model-type engine \
  --trt \
  --load-outputs 06a_polygraphy_precision_alignment/outputs/onnxrt_outputs.json \
  --data-loader-script 06a_polygraphy_precision_alignment/load_npy_input.py \
  --save-outputs 06a_polygraphy_precision_alignment/outputs/trt_compare_outputs.json \
  --rtol 0.001 \
  --atol 0.001 \
  --log-file 06a_polygraphy_precision_alignment/outputs/compare_onnxrt_trt.log \
  --log-format no-colors
```

When `--trt-mode build` is used, Polygraphy builds a temporary TensorRT engine from the ONNX model
and compares ONNX Runtime and TensorRT in one run:

```bash
POLYGRAPHY_INPUT_NPY=05_torch_to_onnx/outputs/input_nchw_float32.npy \
POLYGRAPHY_INPUT_NAME=images \
polygraphy run 05_torch_to_onnx/outputs/yolov8n.onnx \
  --onnxrt \
  --trt \
  --trt-min-shapes images:[1,3,640,640] \
  --trt-opt-shapes images:[1,3,640,640] \
  --trt-max-shapes images:[1,3,640,640] \
  --data-loader-script 06a_polygraphy_precision_alignment/load_npy_input.py \
  --save-outputs 06a_polygraphy_precision_alignment/outputs/trt_compare_outputs.json \
  --rtol 0.001 \
  --atol 0.001 \
  --log-file 06a_polygraphy_precision_alignment/outputs/compare_onnxrt_trt.log \
  --log-format no-colors
```

`--rtol`, `--atol`, `--input-name`, `--input-npy`, `--engine`, and `--output-dir` replace the values
shown above when those options are passed to `align_precision.py`.
