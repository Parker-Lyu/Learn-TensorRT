# 05 - Torch To ONNX

This lesson exports YOLOv8n from PyTorch to a simplified ONNX graph and validates that ONNX Runtime
produces the same raw model output as PyTorch for the same preprocessed tensor.

Goal: build a trustworthy, TensorRT-ready ONNX artifact before using `trtexec` and TensorRT C++
runtime code.

Topics:

- Ultralytics YOLO export
- ONNX opset
- Static shape and dynamic shape
- ONNX Runtime validation
- `onnxsim` or Ultralytics graph simplification
- Netron graph inspection
- Input and output tensor names
- Raw output comparison before NMS

## Why This Matters

TensorRT starts from an ONNX graph. If the ONNX model is already wrong, the TensorRT engine will
only make the wrong result faster.

This lesson validates the boundary between training-framework code and deployment-framework code:

```text
YOLOv8n .pt weights
  -> export to simplified ONNX
  -> inspect input/output names and shapes
  -> run ONNX Runtime on the same NCHW float32 tensor as PyTorch
  -> compare raw [1, 84, 8400] outputs before decode and NMS
```

The comparison happens before postprocessing so later lessons can separate model-export problems
from YOLO decode, NMS, coordinate mapping, TensorRT precision, or C++ buffer bugs.

## Directory Layout

- `export_yolov8_onnx.py`: downloads or uses `assets/yolov8n.pt`, then exports simplified ONNX
  artifacts by default.
- `inspect_onnx.py`: checks the ONNX model and records tensor names, shapes, opset, and operator
  inventory.
- `validate_onnx_runtime.py`: runs PyTorch and ONNX Runtime on the same preprocessed image tensor.
- `outputs/`: generated simplified ONNX models, optional raw ONNX comparisons, tensor dumps, and
  validation reports. This folder is ignored by git.
- `../assets/yolov8n.pt`: shared YOLOv8n weights used by later lessons.
- `../assets/img.jpeg`: shared sample image.

## Export

Run the lesson commands from its directory:

```bash
cd 05_torch_to_onnx
```

Export a simplified static-shape ONNX model:

```bash
python3 export_yolov8_onnx.py
```

The default command writes:

```text
outputs/yolov8n.onnx
```

Export a simplified dynamic-shape ONNX model for later TensorRT optimization-profile experiments:

```bash
python3 export_yolov8_onnx.py --dynamic
```

The dynamic command writes:

```text
outputs/yolov8n_dynamic.onnx
```

Graph simplification is enabled by default because lesson 06 uses these files as the canonical
TensorRT build inputs. Export an unsimplified graph only when you want to compare tool behavior or
debug a simplifier issue:

```bash
python3 export_yolov8_onnx.py --no-simplify --output outputs/yolov8n_raw.onnx
```

The default simplified artifacts are the handoff contract to lesson 06:

```text
outputs/yolov8n.onnx
outputs/yolov8n_dynamic.onnx
```

The default weights path is:

```text
../assets/yolov8n.pt
```

If it is missing, the script downloads the official YOLOv8n weights into the root `assets` folder so
later lessons can reuse the same file.

## Inspect

Check the simplified static model and write a JSON report:

```bash
python3 inspect_onnx.py
```

The report is written to:

```text
outputs/onnx_inspection.json
```

For YOLOv8n static export, the important tensors should look like:

```text
input:  images  FLOAT [1, 3, 640, 640]
output: output0 FLOAT [1, 84, 8400]
```

The `84` dimension is `4` box values plus `80` COCO class scores. The `8400` dimension is the number
of candidate boxes across the detection feature maps.

Open the ONNX file in Netron and confirm:

- The model has one image input.
- The output tensor name and shape match the inspection report.
- The graph does not contain unexpected preprocessing or NMS nodes.

Inspect the dynamic model separately when preparing for TensorRT optimization profiles:

```bash
python3 inspect_onnx.py \
  --onnx outputs/yolov8n_dynamic.onnx \
  --report outputs/onnx_dynamic_inspection.json
```

## Validate

Compare PyTorch raw output with the simplified static ONNX Runtime raw output:

```bash
python3 validate_onnx_runtime.py
```

The script uses the same preprocessing convention as the C++ preprocessing lesson:

```text
BGR image
  -> letterbox to 640 x 640 with padding value 114
  -> BGR to RGB
  -> normalize to float32 [0, 1]
  -> HWC to CHW
  -> add batch dimension, shape [1, 3, 640, 640]
```

The validation command writes:

- `outputs/input_nchw_float32.npy`
- `outputs/pytorch_raw_output.npy`
- `outputs/onnxruntime_raw_output.npy`
- `outputs/validation_preview.txt`
- `outputs/validation_report.json`

Use a different image:

```bash
python3 validate_onnx_runtime.py --image /path/to/image.jpg
```

Use a different tolerance:

```bash
python3 validate_onnx_runtime.py --rtol 1e-3 --atol 1e-3
```

The command exits with status `2` if the outputs are not close enough. That is intentional: a failed
validation should stop the deployment chain until the mismatch is understood.

## Expected Report Fields

`validation_report.json` records:

- weights path
- ONNX path
- image path
- letterbox scale and padding
- ONNX Runtime input/output names
- output shape and dtype
- max absolute error
- mean absolute error
- P99 absolute error
- index and values at the largest mismatch
- `np.allclose` result

These fields become the first accuracy-alignment evidence before Polygraphy and TensorRT enter the
project.

## Checkpoints

- Export both simplified static and dynamic ONNX models and compare their input shapes in the
  inspection reports.
- Change `--imgsz` to `320` and explain why the output candidate count changes.
- Open the model in Netron and find the first `Conv` node and the final output tensor.
- Delete `outputs/yolov8n.onnx`, rerun the export, and confirm the command is reproducible.
- Export `outputs/yolov8n_raw.onnx` with `--no-simplify` and compare its node count with the
  simplified graph.
- Intentionally use a loose and a strict tolerance in validation and explain the difference between
  acceptable float drift and a real export bug.

Acceptance criteria:

- Simplified `outputs/yolov8n.onnx` and `outputs/yolov8n_dynamic.onnx` are generated.
- Input and output tensor names are recorded.
- ONNX checker passes.
- Static and dynamic inspection reports show the expected static tensor shapes and dynamic
  `batch`/`height`/`width` axes.
- ONNX Runtime output from the simplified static ONNX is compared with PyTorch output on the same
  image tensor.
- The validation report explains the numerical difference and tolerance.

## Appendix: Interpreting PyTorch And ONNX FP32 Differences

### Why FP32 Results Are Not Bitwise Identical

Running PyTorch FP32 and ONNX Runtime FP32 on the same CPU does not guarantee identical bits. The
frameworks may use different convolution or matrix-multiplication kernels, SIMD instructions,
threading strategies, reduction orders, graph rewrites, and operator fusion. Because floating-point
addition is not associative, mathematically equivalent execution orders can differ in their last
few digits.

Small drift is therefore expected. Large, widespread, or task-changing differences are not normal
rounding noise and should block the deployment chain until they are understood.

### How The Tolerance Is Applied

This lesson uses NumPy's element-wise closeness rule, followed by `np.allclose`:

```text
abs(pytorch - onnx) <= atol + rtol * abs(onnx)
```

`atol` protects comparisons near zero, where relative error is unstable. `rtol` scales the allowed
difference for larger values. Consequently, `max_abs_error` being greater than `atol` does not by
itself mean validation failed.

For example, an absolute difference of `0.0015` is tiny relative to a box coordinate near `470`,
but the same difference may be important for a confidence score near zero. Always interpret an
error together with the value's scale and meaning.

### Practical FP32 Starting Points

There is no universal tolerance that proves an export is correct. For PyTorch FP32 versus ONNX
Runtime FP32 on the same CPU and with exactly the same input tensor, these ranges are useful
diagnostic starting points:

| Metric or gate | Common starting point | Interpretation |
| --- | --- | --- |
| Strict element-wise gate | `rtol=1e-4`, `atol=1e-5` | A useful first attempt for a well-behaved FP32 graph. |
| Lesson's general gate | `rtol=1e-3`, `atol=1e-3` | More tolerant of mixed-scale raw detector outputs; intentionally not a universal production limit. |
| Mean absolute error | Often `1e-6` to `1e-4` | Describes overall drift, but can hide isolated failures. |
| P99 absolute error | Often `1e-5` to `1e-3` | Describes the typical upper tail without being dominated by one outlier. |
| Maximum absolute error | May reach `1e-3` to `1e-2` when outputs are in the hundreds | Must be interpreted relative to the value at the maximum-error index. |

These are experience-based ranges, not acceptance guarantees. A single `0.01` error on a value near
`500` may be harmless, while `0.01` on a score near `0.001` is large. Numerically sensitive models,
different operators, and different CPU backends may require justified model-specific limits.

For the committed lesson evidence, the raw output shape is `[1, 84, 8400]` and the report records
approximately:

```text
max_abs_error:   1.56e-3
mean_abs_error:  1.43e-6
p99_abs_error:   4.58e-5
close_fraction:  1.0
allclose:        true  (rtol=1e-3, atol=1e-3)
```

The maximum difference occurs on a value near `470.675`, where its relative size is about `3.3e-6`.
Together with the small mean and P99 errors and a `close_fraction` of `1.0`, this is good FP32
alignment rather than a large mismatch.

### Read The Metrics Together

- **Maximum absolute error** finds the worst element, but one outlier can dominate it.
- **Mean absolute error** describes overall drift, but millions of good values can hide a small set
  of bad values.
- **P99 absolute error** shows whether error is widespread in the upper tail.
- **Close fraction** reports the proportion of elements satisfying the element-wise tolerance.
- **`allclose`** is the automated gate: every element must satisfy the configured rule.
- **The values and index at maximum error** reveal whether the mismatch affects a coordinate,
  confidence score, or another output with a different numerical scale.

Also check shape, dtype, and the presence of `NaN` or `Inf` before interpreting error statistics.
Maximum relative error alone is not reliable near zero.

### What Counts As A Suspicious Difference

Investigate rather than immediately relaxing the tolerance when any of the following occurs:

- many elements fail `rtol=1e-3`, `atol=1e-3`;
- `close_fraction` is materially below `1.0`;
- mean or P99 error increases by orders of magnitude compared with a known-good export;
- output contains `NaN` or `Inf`;
- errors are systematic in one channel, spatial region, or operator output;
- repeated runs of the same backend do not reproduce the same result;
- class IDs, confidence threshold decisions, box coordinates, or detection counts change
  materially.

An absolute threshold alone cannot identify every bad export. The distribution and downstream
effect of the error matter more than one unscaled number.

### Triage Procedure For A Failed Comparison

Do not make `rtol` and `atol` progressively looser until the test passes. Use this sequence to find
the cause:

1. **Confirm exact input parity.** Reuse `outputs/input_nchw_float32.npy` and verify image resize,
   letterbox rounding, padding, BGR-to-RGB conversion, normalization, layout, dtype, and batch
   dimension. Do not let each backend preprocess independently during numerical diagnosis.
2. **Confirm inference mode.** Use `model.eval()` and `torch.inference_mode()` so training state,
   Dropout, and BatchNorm updates cannot affect the comparison.
3. **Confirm output semantics.** Check shape, layout, output ordering, and whether both sides expose
   raw logits or decoded/NMS results. Equal element counts do not prove equal semantics.
4. **Check repeatability.** Run PyTorch twice and ONNX Runtime twice with the saved input. If one
   backend disagrees with itself, investigate randomness, mutable state, threading, or an
   uninitialized value first.
5. **Separate export from simplification.** Compare PyTorch with an ONNX model exported using
   `--no-simplify`, then compare the raw and simplified ONNX outputs. A failure introduced only by
   simplification narrows the problem to graph rewriting or constant folding.
6. **Disable ONNX Runtime graph optimization temporarily.** Create a session with
   `ort.GraphOptimizationLevel.ORT_DISABLE_ALL`. If this removes the mismatch, investigate an
   optimization or fusion rather than changing the model-wide tolerance. This is a diagnostic step,
   not necessarily the production configuration.
7. **Inspect sensitive operators.** Pay particular attention to `Resize` coordinate modes and
   rounding, padding, reductions, normalization, division near zero, `Exp`, `Log`, `Softmax`,
   `Sigmoid`, and detection-box decoding.
8. **Locate the first divergent intermediate tensor.** Add selected ONNX intermediates as outputs
   and capture matching PyTorch module outputs with forward hooks. Bisect the graph until the first
   meaningful divergence is found; later layers may only amplify the original discrepancy.
9. **Fix or justify the cause.** Correct preprocessing or output semantics, change an incompatible
   export pattern or opset when necessary, or document a proven backend-level floating-point drift.
   Only then establish a model-specific tolerance.

### Tensor Agreement Is Necessary But Not Sufficient

One image and one `allclose` result are smoke-test evidence, not a production accuracy claim. Run
validation over a representative dataset and examine error distributions by output channel and
input type. For object detection, also compare postprocessed class IDs, confidence scores,
detection counts, box coordinates and IoU, and dataset-level precision, recall, and mAP.

A small raw-tensor difference can cross a confidence or NMS threshold and change a detection. The
opposite is also possible: a visible raw-tensor difference may have no material task-level effect.
Production acceptance should therefore combine reproducible tensor-level tolerances with explicit
task-level quality limits.
