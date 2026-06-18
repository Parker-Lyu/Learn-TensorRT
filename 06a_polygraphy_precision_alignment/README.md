# 06a - Polygraphy Precision Alignment

This lesson uses Polygraphy to compare ONNX Runtime and TensorRT outputs with the exact same
preprocessed YOLOv8n input tensor.

Goal: create a repeatable accuracy-debug workflow for deciding whether backend differences are
acceptable numerical drift or a deployment bug.

Topics:

- Polygraphy model inspection
- ONNX Runtime versus TensorRT comparison
- Saving input and output tensors
- FP32, FP16, and INT8 drift analysis
- Absolute and relative tolerance selection
- First-mismatch debugging workflow
- Reproducible command logs for benchmark reports

## Why This Matters

TensorRT deployment is not finished when an engine builds. The engine must still produce outputs
that match the validated ONNX model closely enough for the target task.

This lesson keeps the comparison narrow and honest:

```text
lesson 05 preprocessed tensor
  -> Polygraphy input JSON
  -> ONNX Runtime output
  -> TensorRT output
  -> error summary and precision note
```

The raw YOLO output is compared before decode, NMS, visualization, or coordinate mapping. That makes
it easier to tell whether drift starts in model execution or in later postprocessing code.

## Directory Layout

- `make_polygraphy_inputs.py`: converts `05_torch_to_onnx/outputs/input_nchw_float32.npy` into
  Polygraphy input JSON.
- `align_precision.py`: runs Polygraphy inspection and inference commands, saves logs, and writes a
  compact precision report.
- `polygraphy_cli_compat.py`: local launcher that keeps Polygraphy working with the repository's
  NumPy 2.x environment without changing system packages.
- `outputs/`: generated Polygraphy inputs, runner outputs, logs, JSON reports, and Markdown notes.
  This folder is ignored by git.
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

## Create Polygraphy Inputs

Convert the lesson 05 NCHW tensor dump into Polygraphy's JSON format:

```bash
python3 06a_polygraphy_precision_alignment/make_polygraphy_inputs.py
```

The default command writes:

```text
06a_polygraphy_precision_alignment/outputs/input_data.json
```

Override the input tensor name if the ONNX inspection report shows a different name:

```bash
python3 06a_polygraphy_precision_alignment/make_polygraphy_inputs.py --input-name images
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
Input:  06a_polygraphy_precision_alignment/outputs/input_data.json
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
mean error, P99 error, close fraction, and the index of the largest mismatch. Decide whether the
detection results remain acceptable for the deployment target before calling the drift acceptable.

## Expected Report Fields

`precision_report.json` records:

- ONNX path
- engine path
- input JSON path
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

Acceptance criteria:

- Polygraphy can run the YOLO ONNX model with ONNX Runtime.
- Polygraphy can run the same model or engine with TensorRT.
- ONNX Runtime and TensorRT outputs are compared using the same input tensor.
- Any mismatch is summarized with max error, mean error, tolerance, and likely cause.
- The final note explains whether the observed drift is acceptable for the deployment target.
