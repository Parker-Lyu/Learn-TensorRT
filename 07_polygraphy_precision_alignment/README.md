# 07 - Polygraphy Precision Alignment

## Purpose

- Learn a repeatable single-input precision-debug workflow when ONNX Runtime and TensorRT outputs
  disagree.
- Real deployment work is not finished when an engine builds successfully.
- Senior candidates should be able to prove where numerical drift starts instead of guessing
  whether preprocessing, export, precision mode, or TensorRT parsing caused the issue.
- A one-image tensor comparison is a debugging gate, not a dataset-level release criterion. Later
  lessons extend it into multi-image drift statistics and decoded detection-quality comparison.

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

Lesson 14 extends this idea when comparing FP32, FP16, and INT8 engines. Lesson 31 should include
both this precision-alignment note and later accuracy-regression evidence.

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

## Deliverables

- `align_precision.py` controlled comparison workflow
- Saved Polygraphy logs and backend outputs
- `precision_report.json` and generated precision-alignment note

## Directory Layout

- `load_npy_input.py`: Polygraphy data loader that feeds the lesson 05 NCHW `.npy` tensor.
- `align_precision.py`: runs Polygraphy inspection and inference commands, saves logs, and writes a
  compact precision report.
- `polygraphy_cli_compat.py`: local launcher that keeps Polygraphy working with the repository's
  NumPy 2.x environment without changing system packages.
- `outputs/`: generated runner outputs, logs, JSON reports, and Markdown notes. This folder is
  ignored by git.
- `../assets/img.jpeg`: canonical image used to generate the controlled lesson 05 input tensor.
- `../05_torch_to_onnx/outputs/yolov8n.onnx`: validated ONNX model from lesson 05.
- `../06_trtexec_engine/outputs/yolov8n_static_fp32.engine`: default serialized TensorRT engine
  from lesson 06.

## Tolerance Notes

Start strict for FP32:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py --rtol 1e-3 --atol 1e-3
```

For FP16, expect larger drift:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py \
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

## Polygraphy Command Reference

`align_precision.py` calls Polygraphy through `polygraphy_cli_compat.py`, which applies the local
NumPy compatibility patch and then forwards arguments to Polygraphy. The displayed command in the
lesson logs starts with `polygraphy`, but the actual Python launcher is:

```bash
python3 07_polygraphy_precision_alignment/polygraphy_cli_compat.py ...
```

The default ONNX inspection command is:

```bash
polygraphy inspect model 05_torch_to_onnx/outputs/yolov8n.onnx \
  --model-type onnx \
  --show layers \
  --log-file 07_polygraphy_precision_alignment/outputs/inspect_onnx.log \
  --log-format no-colors
```

This command does not run inference. It asks Polygraphy to parse the ONNX file and print model
structure information:

- `inspect model`: inspect a model artifact instead of executing it.
- `05_torch_to_onnx/outputs/yolov8n.onnx`: the ONNX model exported and validated in lesson 05.
- `--model-type onnx`: tells Polygraphy how to interpret the file.
- `--show layers`: includes layer-level details, which helps confirm tensor names and shapes.

The ONNX Runtime smoke run uses the `.npy` input through the lesson data loader:

```bash
POLYGRAPHY_INPUT_NPY=05_torch_to_onnx/outputs/input_nchw_float32.npy \
POLYGRAPHY_INPUT_NAME=images \
polygraphy run 05_torch_to_onnx/outputs/yolov8n.onnx \
  --onnxrt \
  --data-loader-script 07_polygraphy_precision_alignment/load_npy_input.py \
  --save-outputs 07_polygraphy_precision_alignment/outputs/onnxrt_outputs.json \
  --log-file 07_polygraphy_precision_alignment/outputs/run_onnxrt.log \
  --log-format no-colors
```

This command runs only the ONNX Runtime backend and saves its raw output tensors:

- `POLYGRAPHY_INPUT_NPY`: path consumed by `load_npy_input.py`; this keeps the controlled input in
  NumPy's binary `.npy` format.
- `POLYGRAPHY_INPUT_NAME`: model input tensor name used in the feed dictionary, normally `images`.
- `run 05_torch_to_onnx/outputs/yolov8n.onnx`: execute the ONNX model.
- `--onnxrt`: enables the ONNX Runtime runner.
- `--data-loader-script`: points Polygraphy to the Python function that yields input tensors.
- `--save-outputs`: writes the runner output in Polygraphy's JSON output format for later loading
  and reporting.

When `--trt-mode engine` is used, the serialized TensorRT engine is inspected first:

```bash
polygraphy inspect model 06_trtexec_engine/outputs/yolov8n_static_fp32.engine \
  --model-type engine \
  --show layers \
  --log-file 07_polygraphy_precision_alignment/outputs/inspect_engine.log \
  --log-format no-colors
```

This command checks the serialized TensorRT artifact before comparison:

- `06_trtexec_engine/outputs/yolov8n_static_fp32.engine`: the engine built in lesson 06.
- `--model-type engine`: tells Polygraphy this file is a serialized TensorRT engine, not ONNX.
- `--show layers`: prints engine layer details when available, useful for confirming the expected
  artifact is being compared.

The default engine comparison command reuses the saved ONNX Runtime output as the reference and
feeds the same `.npy` input to TensorRT:

```bash
POLYGRAPHY_INPUT_NPY=05_torch_to_onnx/outputs/input_nchw_float32.npy \
POLYGRAPHY_INPUT_NAME=images \
polygraphy run 06_trtexec_engine/outputs/yolov8n_static_fp32.engine \
  --model-type engine \
  --trt \
  --load-outputs 07_polygraphy_precision_alignment/outputs/onnxrt_outputs.json \
  --data-loader-script 07_polygraphy_precision_alignment/load_npy_input.py \
  --save-outputs 07_polygraphy_precision_alignment/outputs/trt_compare_outputs.json \
  --rtol 0.001 \
  --atol 0.001 \
  --log-file 07_polygraphy_precision_alignment/outputs/compare_onnxrt_trt.log \
  --log-format no-colors
```

This command runs the serialized TensorRT engine and compares it against the saved ONNX Runtime
reference:

- `run 06_trtexec_engine/outputs/yolov8n_static_fp32.engine`: execute the serialized engine.
- `--model-type engine`: interprets the input artifact as a TensorRT engine.
- `--trt`: enables the TensorRT runner.
- `--load-outputs`: loads the ONNX Runtime outputs saved by the earlier smoke run, so Polygraphy can
  compare TensorRT output against that reference.
- `--data-loader-script`: feeds the exact same `.npy` input tensor to TensorRT.
- `--save-outputs`: saves the TensorRT runner output and comparison artifact.
- `--rtol` and `--atol`: relative and absolute tolerances used by Polygraphy's elementwise
  comparison.

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
  --data-loader-script 07_polygraphy_precision_alignment/load_npy_input.py \
  --save-outputs 07_polygraphy_precision_alignment/outputs/trt_compare_outputs.json \
  --rtol 0.001 \
  --atol 0.001 \
  --log-file 07_polygraphy_precision_alignment/outputs/compare_onnxrt_trt.log \
  --log-format no-colors
```

This command is useful when a serialized lesson 06 engine is not available:

- `run 05_torch_to_onnx/outputs/yolov8n.onnx`: starts from the ONNX model instead of an engine file.
- `--onnxrt` and `--trt`: runs both backends in the same Polygraphy invocation.
- `--trt-min-shapes`, `--trt-opt-shapes`, and `--trt-max-shapes`: define the TensorRT optimization
  profile for the static YOLO input shape. The `images` prefix must match the model input name.
- `--data-loader-script`: still feeds the same controlled `.npy` input tensor.
- `--save-outputs`, `--rtol`, and `--atol`: save runner results and apply the same numerical
  tolerance policy as the serialized-engine path.

`--rtol`, `--atol`, `--input-name`, `--input-npy`, `--engine`, and `--output-dir` replace the values
shown above when those options are passed to `align_precision.py`.

Common logging options:

- `--log-file`: stores Polygraphy's detailed console output in `outputs/` so the lesson can keep a
  reproducible debug record.
- `--log-format no-colors`: removes terminal color codes from logs, making them easier to search and
  include in reports.

## Run

### Input Tensor

This lesson directly reuses the controlled input tensor saved by lesson 05 from `assets/img.jpeg`:

```text
05_torch_to_onnx/outputs/input_nchw_float32.npy
```

`align_precision.py` passes this `.npy` file to Polygraphy through `load_npy_input.py` and
`--data-loader-script`. The tensor remains in NumPy's binary format; no intermediate input JSON is
generated.

Use a different input tensor when experimenting with another preprocessed sample:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py \
  --input-npy path/to/input_nchw_float32.npy \
  --skip-trt
```

Override the input tensor name if the ONNX inspection report shows a different name:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py --input-name images --skip-trt
```

### Smoke Test ONNX Runtime

Run only the ONNX Runtime side when you want to verify Polygraphy setup before using TensorRT:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py --skip-trt
```

This writes:

- `outputs/inspect_onnx.log`
- `outputs/run_onnxrt.log`
- `outputs/onnxrt_outputs.json`
- `outputs/precision_report.json`
- `outputs/precision_alignment_note.md`

Compare the lesson 05 ONNX model against the lesson 06 FP32 engine:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py
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
python3 07_polygraphy_precision_alignment/align_precision.py \
  --engine 06_trtexec_engine/outputs/yolov8n_static_fp16.engine \
  --rtol 1e-2 \
  --atol 1e-2 \
  --keep-going
```

`--keep-going` is useful for FP16 or INT8 experiments because Polygraphy may return a nonzero status
when tolerance fails, but the mismatch evidence is still valuable.

Let Polygraphy build a temporary TensorRT engine from ONNX when a serialized engine is not available:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py --trt-mode build
```

The serialized lesson 06 engine is preferred for normal course work because it compares the exact
artifact that later C++ lessons will load.

## Outputs

- The runnable commands above produce the files and console evidence described in `Deliverables`.
- Generated build and runtime artifacts remain in the lesson's ignored build or output directory.

## Checkpoints

- Run `--skip-trt` and confirm Polygraphy can execute the ONNX model with the saved input tensor.
- Compare FP32 ONNX Runtime and FP32 TensorRT output, then explain the largest mismatch.
- Repeat with the FP16 engine and explain why a different tolerance may be reasonable.
- Intentionally set `--rtol 1e-8 --atol 1e-8` and inspect the generated mismatch evidence.
- Confirm the ONNX model and TensorRT engine came from the same export before debugging
  postprocessing.
- Save the final `precision_alignment_note.md` as evidence for lesson 31.
- Explain why a single-input allclose result is useful for debugging but insufficient for release
  approval.
