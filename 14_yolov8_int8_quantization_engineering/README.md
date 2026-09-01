# 14 - YOLOv8 INT8 Quantization Engineering

## Purpose

This lesson uses `nvcr.io/nvidia/pytorch:25.11-py3` with TensorRT 10.14.1.48 and CUDA 13.0 to
produce a reproducible YOLOv8 deployment decision. The primary workflow is post-training
quantization with explicit ONNX `QuantizeLinear`/`DequantizeLinear` (Q/DQ) nodes. Every precision
conclusion uses the same data, preprocessing contract, postprocessing, evaluator, and predeclared
quality thresholds.

## Prerequisites

- Complete Lesson 05 and Lesson 10. This lesson reuses Lesson 05's ONNX export workflow and
  `10_yolov8_trt_python/infer_yolov8_trt.py` for preprocessing, TensorRT execution, and decoding.
- Lesson 06 is recommended beforehand for its `trtexec` build and benchmarking concepts, but no
  Lesson 06 generated artifact is required.
- Use the shared development environment configured in Course 00; this lesson does not create a
  second course container.
- Ensure network access and sufficient disk space for the documented COCO data; the first command
  in `Run` performs this preparation.

## Deliverables

- Versioned experiment, environment, quality, calibration, and dataset contracts
- ModelOpt export, TensorRT build, precision-audit, validation, and benchmark tools
- Reference-bundle, preprocessing-parity, evaluator, manifest, and contract tests
- Concise generated quantization-run summary
- `docs/reproduction.md` end-to-end reproduction procedure

## Learning Goals

1. Download and identify the COCO calibration and validation data with immutable manifests and
   SHA-256 hashes.
2. Evaluate PyTorch FP32, PyTorch FP16, TensorRT FP32, and TensorRT FP16 references on the complete
   validation split.
3. Export a Q/DQ graph with ModelOpt and build an INT8 engine as a TensorRT 10.14 strongly typed
   network.
4. Accept or reject INT8 with predeclared mAP50-95, mAP50, precision, and recall gates.
5. Benchmark only candidates that pass the quality gates, using matched runtime and measurement
   settings.

## Quality Contract

`configs/quality_contract.json` fixes the input shape, postprocessing behavior, metric
implementation, and thresholds. Do not change thresholds after seeing the result. Any change to a
manifest, model, preprocessing contract, evaluator, runtime identity, or engine requires rebuilding
and reevaluating every affected artifact.

## Run

Run the following commands from the repository root in the shared development environment from
Course 00. The complete ordered procedure is also available in
[`docs/reproduction.md`](docs/reproduction.md). Prepare and qualify the data first:

```bash
python3 assets/coco/prepare_coco.py
python3 14_yolov8_int8_quantization_engineering/tools/prepare_calibration_dataset.py --materialize
python3 14_yolov8_int8_quantization_engineering/tools/analyze_calibration_representativeness.py
python3 14_yolov8_int8_quantization_engineering/tools/verify_preprocessing_parity.py
```

Export the static course ONNX model, establish the four references, then export, build, evaluate,
and inspect the Q/DQ INT8 candidate:

```

<details><summary>Example output (local run)</summary>

```text
run summary written to outputs/
```
</details>
bash
python3 05_torch_to_onnx/export_yolov8_onnx.py
python3 05_torch_to_onnx/inspect_onnx.py
python3 05_torch_to_onnx/validate_onnx_runtime.py
python3 14_yolov8_int8_quantization_engineering/modelopt/export_qdq.py \
  --high-precision fp16 --name yolov8n_qdq_fp16
python3 14_yolov8_int8_quantization_engineering/modelopt/build_engines.py
python3 14_yolov8_int8_quantization_engineering/compare_engines.py \
  --experiment-id modelopt_qdq_int8
python3 14_yolov8_int8_quantization_engineering/modelopt/inspect_precision.py
python3 14_yolov8_int8_quantization_engineering/modelopt/validate_outputs.py
```

`compare_engines.py` produces unified JSON and Markdown results for the same validation split. INT8
must pass gates relative to both PyTorch FP32 and TensorRT FP16; a failing candidate is excluded
from the performance recommendation.

Collect the canonical matched runtime evidence and generate the concise Lesson 14 execution summary:

```bash
python3 14_yolov8_int8_quantization_engineering/modelopt/benchmark_engines.py
python3 14_yolov8_int8_quantization_engineering/tools/generate_run_summary.py
```

The benchmark tool verifies that the engines are the same artifacts used by `compare_engines.py`.
It always measures FP32 and FP16, but measures INT8 only when the INT8 backend passed its
predeclared quality gate. A failing INT8 candidate is recorded as rejected rather than benchmarked.

Lesson 14 writes machine-readable evaluation, inspection, and performance evidence under its
ignored `outputs/` directory. It also writes
`outputs/summary/quantization_run_summary.md`, a short execution summary rather than an
application-facing deployment report. Lesson 15 validates the same evidence and generates the
application-facing decision report; Lesson 14 does not generate a root-level course report.

## Outputs

- Environment-specific engines, timing caches, predictions, and intermediate evidence are written
  under ignored `outputs/`.
- `outputs/summary/quantization_run_summary.md` summarizes the run and remains ignored with the
  evidence from which it was generated.
- Lesson 15's generated `reports/15_precision_performance.md` is ignored and must be regenerated
  for the current environment.

## Tests

Run both CPU-only suites from the repository root:

```bash
PYTHONPATH=14_yolov8_int8_quantization_engineering \
python3 -m unittest discover -s 14_yolov8_int8_quantization_engineering/tests -v
PYTHONPATH=14_yolov8_int8_quantization_engineering \
python3 -m unittest discover -s 14_yolov8_int8_quantization_engineering/modelopt -p 'test_*.py' -v
```

These tests validate contracts and runtime-independent logic. They do not replace TensorRT, CUDA,
PyTorch, ModelOpt, engine, or dataset-level validation in the pinned GPU container.

## Checkpoints

1. Build matched FP32 and FP16 references before evaluating ModelOpt explicit-Q/DQ INT8.
2. Enforce immutable dataset, preprocessing, evaluator, environment, and quality contracts.
3. Audit actual TensorRT layer precision and make a deployment decision from saved quality and performance evidence.

## Appendix: Default TensorRT Engine Build Commands

With no command-line arguments, `modelopt/build_engines.py` invokes `trtexec` three times in FP32,
FP16, and INT8-candidate order. The script resolves every artifact path to an absolute path. The
following commands show the same argument vectors in a portable form; run them from the repository
root and let the shell expand `REPO_ROOT`:

```bash
REPO_ROOT="$(pwd)"
mkdir -p \
  "${REPO_ROOT}/14_yolov8_int8_quantization_engineering/outputs/tensorrt10/references" \
  "${REPO_ROOT}/14_yolov8_int8_quantization_engineering/outputs/tensorrt10/candidate"

trtexec \
  --onnx="${REPO_ROOT}/05_torch_to_onnx/outputs/yolov8n.onnx" \
  --saveEngine="${REPO_ROOT}/14_yolov8_int8_quantization_engineering/outputs/tensorrt10/references/yolov8n_trt10_fp32.engine" \
  --stronglyTyped \
  --builderOptimizationLevel=3 \
  --skipInference \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --exportLayerInfo="${REPO_ROOT}/14_yolov8_int8_quantization_engineering/outputs/tensorrt10/references/yolov8n_trt10_fp32.layers.json" \
  --timingCacheFile="${REPO_ROOT}/14_yolov8_int8_quantization_engineering/outputs/tensorrt10/references/trt10_reference.timing.cache"

trtexec \
  --onnx="${REPO_ROOT}/05_torch_to_onnx/outputs/yolov8n.onnx" \
  --saveEngine="${REPO_ROOT}/14_yolov8_int8_quantization_engineering/outputs/tensorrt10/references/yolov8n_trt10_fp16.engine" \
  --fp16 \
  --builderOptimizationLevel=3 \
  --skipInference \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --exportLayerInfo="${REPO_ROOT}/14_yolov8_int8_quantization_engineering/outputs/tensorrt10/references/yolov8n_trt10_fp16.layers.json" \
  --timingCacheFile="${REPO_ROOT}/14_yolov8_int8_quantization_engineering/outputs/tensorrt10/references/trt10_reference.timing.cache"

trtexec \
  --onnx="${REPO_ROOT}/14_yolov8_int8_quantization_engineering/outputs/qdq/yolov8n_qdq_fp16.onnx" \
  --saveEngine="${REPO_ROOT}/14_yolov8_int8_quantization_engineering/outputs/tensorrt10/candidate/yolov8n_qdq_int8.engine" \
  --stronglyTyped \
  --builderOptimizationLevel=3 \
  --skipInference \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --exportLayerInfo="${REPO_ROOT}/14_yolov8_int8_quantization_engineering/outputs/tensorrt10/candidate/yolov8n_qdq_int8.layers.json" \
  --timingCacheFile="${REPO_ROOT}/14_yolov8_int8_quantization_engineering/outputs/tensorrt10/candidate/qdq_int8.timing.cache"
```

The wrapper creates the destination directories and redirects each command's combined standard
output and error stream to its corresponding `*.build.log`. It also rejects TensorRT versions other
than `10.14.1.48`, verifies the expected engine I/O contract, and writes artifact hashes and command
metadata. Running the raw commands alone therefore does not reproduce those surrounding checks or
metadata files.

### Wrapper Parameters

The no-argument invocation uses all three parser defaults:

| Parameter | Default | Effect |
| --- | --- | --- |
| `--output-dir` | `outputs/tensorrt10` under this lesson | Selects the artifact root. The wrapper resolves it to an absolute path before constructing commands. |
| `--trtexec` | `trtexec` | Selects the executable. Supply an explicit path when `trtexec` is not on `PATH`. |
| `--only` | `all` | Builds all three engines. `fp32`, `fp16`, or `int8` builds only that engine; a partial run does not write `references/reference_builds.json`. |

### `trtexec` Parameters Used

| Parameter | Engines | Purpose |
| --- | --- | --- |
| `--onnx` | All | Selects the source graph. FP32 and FP16 share the Lesson 05 graph; the INT8 candidate uses the ModelOpt explicit-Q/DQ graph. |
| `--saveEngine` | All | Serializes the environment-specific TensorRT engine to the indicated ignored output path. |
| `--stronglyTyped` | FP32, INT8 candidate | Makes tensor types part of the network contract. For the Q/DQ graph, ONNX Q/DQ operators, rather than a blanket builder precision flag, define the quantized regions. |
| `--fp16` | FP16 | Enables FP16 tactic selection on the ordinary ONNX graph. It permits, but does not force, every layer to execute in FP16. |
| `--builderOptimizationLevel=3` | All | Uses the same builder-search level for comparable engine builds. |
| `--skipInference` | All | Builds and saves the engine without running the `trtexec` inference benchmark. Benchmarking is a separate matched workflow in this lesson. |
| `--profilingVerbosity=detailed` | All | Retains detailed layer metadata for later inspection. |
| `--dumpLayerInfo` | All | Requests layer information in the build output. |
| `--exportLayerInfo` | All | Writes machine-readable layer information to the corresponding `*.layers.json` file. |
| `--timingCacheFile` | All | Loads and updates a timing cache. FP32 and FP16 intentionally share the reference cache; the Q/DQ candidate has a separate cache. |

### Important `trtexec` Parameters Not Used

These omissions are intentional for the fixed Lesson 14 contract, not general recommendations for
every TensorRT build:

| Parameter or group | Why it is absent here | When it becomes important |
| --- | --- | --- |
| `--int8` and `--calib` | The candidate is a strongly typed explicit-Q/DQ graph. Its quantization scales are embedded in ONNX, so this build does not perform implicit builder calibration. Adding `--int8` is not the mechanism that makes this candidate INT8. | A supported weakly typed INT8 workflow or a builder-calibration workflow with an appropriate calibration cache. |
| `--minShapes`, `--optShapes`, `--maxShapes`, and `--shapes` | The course graphs have the fixed input contract `[1, 3, 640, 640]`; no optimization profile or runtime shape needs to be supplied. | Dynamic batch sizes or spatial dimensions. Define all profile bounds deliberately, then benchmark the relevant runtime shapes. |
| `--memPoolSize` | The lesson leaves TensorRT memory-pool limits at their defaults. | Constrained deployment targets or controlled comparisons where the builder workspace limit must be explicit. |
| `--precisionConstraints`, `--layerPrecisions`, and `--layerOutputTypes` | The strongly typed builds take types from the graph, while the FP16 reference deliberately allows TensorRT to choose tactics without per-layer overrides. | Precision-debugging experiments or weakly typed networks that require selected layers to retain a particular precision. |
| `--inputIOFormats` and `--outputIOFormats` | The lesson uses the ONNX/default linear FP32 I/O contract recorded by the wrapper. Quantization is internal to the Q/DQ graph. | Applications with a deliberately different boundary type or tensor format. Update preprocessing, bindings, and validation together. |
| `--tacticSources` | The lesson accepts the TensorRT 10.14 default tactic sources. | Reproducibility investigations, plugin constraints, or target environments that require an explicit tactic-source policy. |
| `--hardwareCompatibilityLevel` and `--versionCompatible` | Engines are built for the current pinned TensorRT/GPU environment and are treated as environment-specific artifacts. | A delivery lesson with an explicit compatibility target and validation matrix; these flags do not make an engine universally portable. |
| `--useCudaGraph`, `--noDataTransfers`, `--useSpinWait`, `--warmUp`, `--duration`, and `--iterations` | `--skipInference` disables `trtexec` benchmarking, and Lesson 14 uses `modelopt/benchmark_engines.py` for matched measurements. | Direct `trtexec` performance experiments, where the complete measurement protocol must be recorded. |

There is no separate FP32-enabling flag in the first command: FP32 tensor types come from the
ordinary ONNX graph under `--stronglyTyped`. Likewise, the name `yolov8n_qdq_int8.engine` describes
the quantized candidate; it does not imply that every layer executes in INT8. Use the exported layer
information and `modelopt/inspect_precision.py` to verify the actual per-layer precision mix.
