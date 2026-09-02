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

Lesson 14 extends this idea when comparing FP32, FP16, and INT8 engines. Lesson 32 should include
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

Run the standard lesson workflow with this controlled tensor (and the default lesson 06 FP32
engine) as follows:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py \
  --input-npy 05_torch_to_onnx/outputs/input_nchw_float32.npy
```

The explicit `--input-npy` makes the source of the comparison input visible in a copied command;
omitting it is equivalent because this path is the script default.

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

<details><summary>Example output (local run, partial)</summary>

```text
== compare_onnxrt_trt ==
report: /workspace/Learn-TensorRT/07_polygraphy_precision_alignment/outputs/precision_report.json
note: /workspace/Learn-TensorRT/07_polygraphy_precision_alignment/outputs/precision_alignment_note.md
allclose(rtol=0.001, atol=0.001): True
max abs error: 0.00146484375
likely cause: within tolerance; keep the evidence with the benchmark report
```
</details>

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

## Appendix: ONNX → TensorRT Precision Alignment and Troubleshooting Notes

After a model is converted from ONNX to TensorRT, its raw outputs may differ numerically from
ONNX Runtime (ORT). The core debugging principle is: **fix the input and model version, establish an
FP32 baseline first, then check FP16 or INT8; first locate where the error starts, then decide how
to fix it.**

This note is a troubleshooting checklist. It does not replace this lesson's single-input report, nor
the multi-sample, decoded-detection, and dataset-level task metrics. A single tensor's `allclose`
result only answers "is this sample's raw output within the given tolerance"; it cannot, by itself,
prove that model precision meets release requirements.

### 1. Core Tools

| Tool | Primary use | Limits |
| --- | --- | --- |
| **Polygraphy** | Run ORT/TRT, save inputs/outputs, compare tensor by tensor, reduce failing subgraphs, and search precision constraints | Marking per-layer outputs can change TensorRT fusion and memory footprint; treat results as localization clues |
| **Netron** | Inspect nodes, tensor names, attributes, and graph structure | Visualization only; does not validate actual runtime semantics |
| **trtexec** | Build and analyze engines, record parse logs, layer info, precision, tactics, and performance | Performance analysis is not a substitute for numerical validation |
| **Polygraphy Surgeon / ONNX GraphSurgeon** | Extract subgraphs, constant folding, and evidence-based graph edits | After editing, re-run ONNX Checker, the ORT baseline, and the TRT comparison |

### 2. Fix the Comparison Conditions First

Many "precision problems" are actually differences in comparison conditions. Confirm at least the
following are consistent:

1. The ONNX model and the TensorRT engine come from the same export; a serialized engine cannot be
   discussed apart from its build environment.
2. Both backends read the exact same preprocessed input tensor, rather than each reading and
   preprocessing the same image separately.
3. Input names, shapes, data types, layout, and dynamic shape profiles are consistent.
4. The comparison uses outputs from the same semantic stage, for example both before decode and NMS.
5. Random seeds, plugin versions, TensorRT/CUDA/GPU/driver, and build options are recorded.
6. Inspect absolute error, relative error, error quantiles, `NaN`/`Inf`, and task-level metrics
   together; do not look only at a single maximum relative error, because that metric is amplified
   when the reference value is near zero.

Recommended timeline:

```text
Confirm the comparison conditions
  → Verify the ORT baseline itself
  → Align ORT and TRT FP32
  → If FP32 fails, locate the parsing, graph-semantics, or tactic problem
  → If FP32 passes, test FP16 or explicit Q/DQ INT8
  → Apply the smallest fix for the localized problem
  → Regress single-input, representative-sample-set, and task-level metrics
```

### 3. Stage One: FP32 Baseline

#### 3.1 Start with this lesson's controlled comparison

First run this lesson's existing workflow, which reuses the `.npy` input saved by lesson 05 and
compares ORT against lesson 06's FP32 engine:

```bash
python3 07_polygraphy_precision_alignment/align_precision.py
```

Do not loosen tolerance just to make the result `PASSED`. First inspect the maximum, mean, and P99
absolute errors, the largest-mismatch position, output shapes, and outliers in the report, then set
tolerances from the business metrics.

#### 3.2 Verify the impact of TF32 separately

On Ampere and newer architectures, TF32 may cause FP32 networks to pick implementations with higher
throughput but different numerical behavior. However, **the Polygraphy 0.49.26 pinned by this course
enables TF32 explicitly with `--tf32`; there is no `--no-tf32` flag**. Therefore, when building a
temporary baseline from ONNX with Polygraphy, first run without `--tf32`, then add that flag
separately for an A/B comparison:

```bash
# Baseline: do not explicitly enable TF32
polygraphy run model.onnx --onnxrt --trt   --data-loader-script your_data_loader.py   --rtol 1e-4 --atol 1e-4

# Control: explicitly enable TF32
polygraphy run model.onnx --onnxrt --trt --tf32   --data-loader-script your_data_loader.py   --rtol 1e-4 --atol 1e-4
```

Different frontends or TensorRT APIs may handle build flags differently by default, so do not
generalize the default behavior of the commands above to every build path. Rely on the actual
commands, BuilderConfig, and build logs. If only the TF32 control fails, the difference is related to
the chosen numerical path, but whether to disable it should still be decided by task-level accuracy
and performance data.

#### 3.3 Find the earliest clear divergence tensor by tensor

After confirming the final output actually fails, temporarily mark intermediate tensors as outputs:

```bash
polygraphy run model.onnx   --onnxrt --trt   --data-loader-script your_data_loader.py   --onnx-outputs mark all   --trt-outputs mark all   --rtol 1e-3 --atol 1e-3
```

Focus on the earliest location along the topological path where error growth clearly appears, rather
than mechanically declaring the first `FAILED` node in the log as the root cause. Branching graphs,
near-zero reference values, node ordering, and error accumulation can all mislead judgment. In
addition, marking all outputs suppresses some fusion and increases memory usage; for large models,
mark tensors in batches around the suspected region and reproduce the final error with the
un-instrumented full network.

#### 3.4 Extract a minimal reproducible subgraph

After finding a suspicious tensor, extract that region. Polygraphy 0.49.26's metadata format is
`name:[shape]:dtype`:

```bash
polygraphy surgeon extract model.onnx   --inputs 'suspect_input:[1,64,80,80]:float32'   --outputs 'suspect_output:float32'   --output single_region.onnx
```

Extracting a subgraph does not automatically prove a specific operator is wrong. To reproduce the
problem in the full network, save and reuse the real inputs that the original network produced at the
subgraph boundary; random inputs may not cover the same numerical range. After extraction, run ONNX
Checker and ORT first, then compare TensorRT with the same inputs.

For large networks you can also use the experimental `debug reduce` to shrink the failing graph
automatically:

```bash
polygraphy debug reduce model.onnx   --output reduced_failing_model.onnx   --model-input-shapes 'images:[1,3,640,640]'   --check polygraphy run polygraphy_debug.onnx     --onnxrt --trt     --data-loader-script your_data_loader.py     --rtol 1e-3 --atol 1e-3
```

`debug reduce` judges whether the current subgraph is good or bad from the check command's exit
status, so first confirm the check command can reproduce the target failure reliably. For dynamic
shapes and shape subgraphs, pin the debug shapes first when necessary; do not mistake the reduced
result for a final fix of the full model.

### 4. Fix Order for FP32 Problems

#### 4.1 Fix the export source first

If the ONNX graph's attributes or semantics are already wrong, first fix the PyTorch export source
and re-export. For example:

- Make the `Resize`/`interpolate` mode and semantics such as `align_corners` explicit;
- Use a reasonable `epsilon` for normalization or division when the algorithm allows;
- Avoid relying on undefined behavior, input-dependent control flow, or operations the exporter
  cannot express reliably.

Do not arbitrarily change operator attributes just to make ORT and TRT align. Changes must match the
original model semantics, and PyTorch and ONNX outputs must be re-validated.

#### 4.2 Do ONNX graph surgery only when re-export is impossible

The example below only demonstrates the mechanics; whether `half_pixel` is correct must be decided by
the original framework's semantics:

```python
import onnx
import onnx_graphsurgeon as gs

graph = gs.import_onnx(onnx.load("model.onnx"))
for node in graph.nodes:
    if node.name == "Resize_45" and node.op == "Resize":
        node.attrs["coordinate_transformation_mode"] = "half_pixel"

graph.cleanup().toposort()
fixed = gs.export_onnx(graph)
onnx.checker.check_model(fixed)
onnx.save(fixed, "model_fixed.onnx")
```

Constant folding can simplify statically computable subgraphs, but it is not a general precision fix:

```bash
polygraphy surgeon sanitize model.onnx   --fold-constants   --output sanitized_model.onnx
```

#### 4.3 Consider plugin or tactic-level diagnosis last

Consider a custom plugin only after confirming that TensorRT 10.14 cannot correctly express or
implement the target semantics and the model cannot be reasonably rewritten. New TensorRT 10.14
plugins should prefer the `IPluginV3` interface; the old `IPluginV2` family and deprecated plugins
should not be the default for new course code. First check whether TensorRT already has a native
layer or a supported ONNX mapping.

If you suspect a specific tactic or fusion path is at fault, save the build log and a reproducible
model, and run controlled experiments with Polygraphy's `debug build`, tactic replay, or precision
constraints. TensorRT has no universal "disable a specific fusion" switch that works for every
network, and tactic sources should not be disabled arbitrarily without evidence.

### 5. Stage Two: FP16 and Explicit Q/DQ INT8

Only after the FP32 baseline passes can you attribute newly introduced drift to the low-precision
path. Test FP16 and INT8 separately; do not change precision, inputs, optimization profiles, and
model version all at once.

#### 5.1 FP16

Check, layer by layer, the region where error first grows clearly, and also check:

- Whether intermediate tensors show `NaN`, `Inf`, overflow, or underflow;
- Numerical ranges around normalization, exponentials, division, and large reductions;
- A layer's computation precision versus its output tensor type — these are not the same concept;
- Whether decoded results and task metrics still meet quality thresholds after loosening tolerance.

Polygraphy can experimentally search for which layers need higher precision. The command needs a
`--check` command that decides pass/fail automatically; omitting it enters interactive mode:

```bash
polygraphy debug precision model.onnx   --fp16   --mode bisect   --precision float32   --check your_accuracy_check_command
```

The search result is a diagnostic clue, not a final design. For weakly-typed networks, verify via
per-layer precision constraints; for strongly-typed networks or explicitly quantized models, adjust
within the bounds allowed by the model type and the quantization graph, and do not assume
`ILayer::setPrecision(kFLOAT)` works for every TensorRT 10.14 network.

#### 5.2 INT8

This course centers on **explicit Q/DQ** models exported with ModelOpt; the complete engineering flow
is in lesson 14. The `IInt8Calibrator`-based implicit quantization flow is deprecated in TensorRT
10.14, so switching `ENTROPY_CALIBRATION_2`/`MINMAX_CALIBRATION` should not be the course's
preferred fix.

For explicit Q/DQ INT8, focus on checking:

1. Whether representative calibration data reuses the exact same preprocessing as deployment;
2. Whether Q/DQ scale, granularity, axes, and symmetry match the target operator and hardware
   constraints;
3. Whether outliers make a few samples dominate the effective quantization range;
4. Whether sensitive layers need to stay FP16/FP32 and are expressed explicitly by the ModelOpt
   configuration and the exported Q/DQ graph;
5. Whether multi-sample tensor drift, decoded detection results, and dataset-level metrics all pass
   at the same time.

There is no universal "100–500 calibration images" answer that is correct for every model. Sample
count is only one variable; distribution coverage, class and scene representativeness, preprocessing
consistency, and final task metrics matter more.

### 6. Quick Reference for Common Symptoms

| Symptom | Check first | Do not conclude prematurely |
| --- | --- | --- |
| FP32 final outputs disagree | Model/input version, shape, TF32 configuration, first clearly drifting tensor, parsing and tactic logs | "It must be a quantization problem" |
| Drift near `Resize` | Mode, coordinate transform, nearest rounding, opset, and original-framework semantics | "Changing to `half_pixel` uniformly will fix it" |
| Drift in `GridSample` or boundary operators | Padding, coordinate normalization, `align_corners`, actual TensorRT 10.14 support | "A third-party plugin is required" |
| NMS result order differs | Near-threshold candidate boxes, equal-score ordering, output set, and task metrics | "The original floating-point network is already distorted" |
| FP16 shows huge error | `NaN`/`Inf`, dynamic range, sensitive reductions/norm/exp, layer precision | "All Norm/Softmax must fall back to FP32" |
| INT8 deep-output distortion | Q/DQ scale, calibration distribution, outliers, preprocessing, sensitive-layer strategy | "Adding more calibration images will fix it" |
| Constant or shape subgraph anomaly | Shape inference, dynamic dimensions, foldable subgraph, and parsing logs | "Constant folding is always safe and fixes precision" |

### 7. Final Decision Tree

```text
ORT and TRT outputs diverge
 ├─ Are the comparison conditions fully consistent?
 │   ├─ No: unify the model, input, shape, output semantics, and build configuration first
 │   └─ Yes: run the FP32 baseline
 ├─ Does FP32 pass the defined numerical threshold?
 │   ├─ No: check TF32 configuration → mark intermediate tensors in batches → save boundary inputs
 │   │      → extract/reduce a reproducible subgraph → fix export semantics, parsing, or a proven tactic issue
 │   └─ Yes: run FP16 or explicit Q/DQ INT8 separately
 ├─ Does low precision introduce unacceptable new drift?
 │   ├─ Yes: locate the first clearly amplified region → check dynamic range / QDQ scale
 │   │      → apply evidence-based mixed precision or re-quantization to the sensitive region
 │   └─ No: proceed to multi-sample validation
 └─ Only after single-input, representative-sample-set, decoded results, and task-level metrics all
     pass do you form release evidence
```

After every fix, re-run the baseline from the full model rather than only validating the extracted
subgraph. The final report should retain the model identity, environment identity, precision
configuration, input source, tolerances, numerical statistics, and task-level validation conclusions.
