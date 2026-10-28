# 06 - FP16 ONNX and TensorRT engines

## Purpose

Build TensorRT engines with `trtexec` before writing a C++ runtime. Compare a strict FP32 baseline,
the deprecated weakly typed `--fp16` compatibility route, and explicit ModelOpt AutoCast ONNX with
`--stronglyTyped`. Learn to interpret dynamic profiles, timing caches, memory, and benchmark evidence.

## Prerequisites

Complete lesson 00 and run the following from the repository root **inside the persistent
`learn-tensorrt` container**, as documented there. Export the static and dynamic models, then save
the validated image tensor used by the conversion gate:

```bash
python3 05_torch_to_onnx/export_yolov8_onnx.py
python3 05_torch_to_onnx/export_yolov8_onnx.py --dynamic
python3 05_torch_to_onnx/validate_onnx_runtime.py
```

These commands produce the two source ONNX graphs and `input_nchw_float32.npy` under lesson 05's
ignored `outputs/` directory. No additional packages are needed beyond the course container.

## Deliverables

- `prepare_fp16_onnx.py`: explicit mixed-precision conversion and CPU ONNX Runtime numerical gate.
- `build_and_benchmark.py`: five-engine matrix, guarded engine reuse, and benchmark evidence.
- `summarize_results.py`: validated evidence report with shapes, memory, and latency percentiles.
- CPU regression tests for gates, cache invalidation, command generation, and evidence parsing.
- Ignored models, engines, build receipts, timing cache, logs, JSON exports, manifest, and report.

## Run

Prepare and validate the explicit mixed-precision graphs. The script retains the detection head in
FP32 and compares source and converted raw outputs on the same saved input:

```bash
python3 06_trtexec_engine/prepare_fp16_onnx.py
```

Example output (recorded in the development container):

```text
static: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/yolov8n_static_autocast_fp16.onnx
validation: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/static_fp16_onnx_validation.json
dynamic: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/yolov8n_dynamic_autocast_fp16.onnx
validation: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/dynamic_fp16_onnx_validation.json
```

Preview all five build commands. This checks source files, successful conversion reports, and
source/converted hashes without constructing engines; it requires `trtexec` in the container PATH:

```bash
python3 06_trtexec_engine/build_and_benchmark.py --dry-run
```

Example output (partial; command lines omitted):

```text
== static_fp32 ==
== static_fp16_legacy ==
== dynamic_fp16_legacy ==
== static_fp16_strong ==
== dynamic_fp16_strong ==
```

Build and benchmark the matrix. Each build uses the existing `outputs/trtexec_timing.cache` when
available. With a matching `.build.json` receipt, model hash, engine hash, build settings, and runtime
environment, subsequent invocations load the saved engine and perform a **fresh benchmark**:

```bash
python3 06_trtexec_engine/build_and_benchmark.py
```

<details><summary>Example output (partial; command lines omitted, initial build)</summary>

```text
== static_fp32 (build with timing cache) ==
== static_fp16_legacy (build with timing cache) ==
== dynamic_fp16_legacy (build with timing cache) ==
== static_fp16_strong (build with timing cache) ==
== dynamic_fp16_strong (build with timing cache) ==
manifest: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/build_manifest.json
```
</details>

A successful run prints the manifest path after all selected engines finish. Previous engine files
without a verified receipt are rebuilt using the timing cache. During this verification, TensorRT
rejected the preexisting cache because its serialization version did not match; the build produced
a fresh compatible cache and later engines loaded it. No incompatible cache was forced into use.

<details><summary>Example output (partial; command lines omitted, repeat run)</summary>

```text
== static_fp32 (reuse engine) ==
== static_fp16_legacy (reuse engine) ==
== dynamic_fp16_legacy (reuse engine) ==
== static_fp16_strong (reuse engine) ==
== dynamic_fp16_strong (reuse engine) ==
manifest: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/build_manifest.json
```
</details>

Generate the report. The summarizer requires a successful log, nonempty engine, valid JSON exports,
and all requested timing statistics for every manifest entry; missing evidence causes failure:

```bash
python3 06_trtexec_engine/summarize_results.py
```

Example output:

```text
report: /workspace/Learn-TensorRT/06_trtexec_engine/outputs/benchmark_summary.md
```

To build only what later lessons need, select the strict FP32 reference or the recommended strongly
typed FP16 engines. Use `--models static` during preparation if only the static FP16 model is needed:

```bash
python3 06_trtexec_engine/build_and_benchmark.py --builds static_fp32
python3 06_trtexec_engine/build_and_benchmark.py --builds static_fp16_strong dynamic_fp16_strong
```

A selected run writes a manifest for that selection. Run the default matrix again to regenerate a
five-engine comparison; matching engines will be reused. FP32-only builds do not require converted
FP16 graphs. Strongly typed builds always require their successful conversion reports, even when
an engine can be reused.

Use `--timing-cache-file /path/to/existing.cache` to share an existing timing cache from the **same
GPU and software environment** with a different output directory. This reduces tactic search; it
does not eliminate all builder work. Use `--rebuild` to force engine construction while still using
the timing cache. Do not force incompatible caches across hardware or TensorRT versions.

Use `--output-dir` for an independent engine/evidence directory. Conversion reports are resolved
beside their converted ONNX graphs, independently of the engine output location. The summarizer's
`--output-dir` also controls its default report location; `--report` overrides the report path.

## Outputs

Committed deliverables are the three scripts, regression tests, and this README. All files below
are generated, environment-specific, and ignored under `06_trtexec_engine/outputs/`:

| Engine | Source / typing | Later-course use |
| --- | --- | --- |
| `yolov8n_static_fp32.engine` | FP32 ONNX, weakly typed, TF32 disabled | Strict reference in 07–13 |
| `yolov8n_static_fp16_strong.engine` | AutoCast ONNX, strongly typed | Recommended static FP16 examples in 07, 08, 10, 11, 13 |
| `yolov8n_dynamic_fp16_strong.engine` | AutoCast ONNX, strongly typed with profile | Dynamic FP16 examples in 08 and 10 |
| `yolov8n_static_fp16_legacy.engine` | FP32 ONNX, weakly typed `--fp16` | Compatibility comparison |
| `yolov8n_dynamic_fp16_legacy.engine` | FP32 ONNX, weakly typed `--fp16` with profile | Compatibility comparison |

The obsolete names `yolov8n_static_fp16.engine` and `yolov8n_dynamic_fp16.engine` are no longer
produced or used by the course. Rebuild named deliverables from this revision rather than renaming
an old engine whose provenance is unknown.

Other generated evidence includes:

- `*_autocast_fp16.onnx`, representative `.npz` inputs, AutoCast logs, and
  `*_fp16_onnx_validation.json` with conversion hashes and numerical results.
- `*.build.json` receipts containing engine identity, builder command, and compatibility metadata.
  A matching receipt allows engine reuse; modifying the model, engine, build profile, or environment
  invalidates it. Changing benchmark duration does not require engine construction.
- `trtexec_timing.cache`, logs, `*_times.json`, `*_layers.json`, and `*_profile.json`.
- `build_manifest.json` recording fresh measurement times, executed commands, input shapes, typing,
  TF32 policy, and runtime/CUDA/GPU/driver/container identity.
- `benchmark_summary.md` separating serialized file size, TensorRT context allocations, total
  latency percentiles, and GPU compute percentiles. Context allocation is not peak process memory.

A failed build does not publish a new manifest. Old per-engine files or an older Markdown report
are not evidence that the current matrix passed; check the manifest and measurement timestamps.

## Tests

Run focused CPU tests in the development container. They need the baseline NumPy/ONNX/ONNX Runtime
packages, but do not invoke a GPU builder or inference runtime:

```bash
python3 -m unittest discover -s 06_trtexec_engine/tests -v
```

The tests cover failed/missing conversion reports, stale source and candidate hashes, independent
model/engine output directories, build-selection conflicts, timing-cache arguments, engine reuse
and invalidation, stale/missing benchmark artifacts, metric-specific percentiles, context memory,
and invalid numerical comparisons. Actual TensorRT engine construction and dynamic-shape execution
require the GPU workflow in `Run`; mocked subprocess tests do not establish GPU correctness.

## Checkpoints

1. What is serialized in an engine, and why should its GPU/software identity be recorded?
2. Explain weak typing, strong typing, and why `--fp16` is a compatibility route rather than a
   guarantee that every layer or I/O binding uses FP16. Inspect the layer-info JSON to check types.
3. Identify the dynamic min/opt/max shapes and actual benchmark shape. Does a successful 640×640
   measurement prove that the minimum and maximum profile shapes ran successfully?
4. Distinguish the workspace limit, serialized file size, TensorRT context allocations, and total
   process GPU memory. Which of these does the report actually measure?
5. Compare latency P99 with GPU compute P99 and H2D/D2H means. Why is qps not generally the
   reciprocal of latency? How does batch size change the interpretation of qps?
6. Change a copy of a converted model or its validation hash and verify that the build driver
   rejects it before invoking `trtexec`. Restore valid artifacts before continuing.
7. Repeat a build command. Confirm that it loads matching engines but refreshes measurement logs.
   Then change a build setting and explain why timing-cache reuse and engine reuse differ.

## Design

An engine contains TensorRT's optimized executable network and selected tactics. An optimization
profile bounds a dynamic input and gives the builder an optimum shape for tactic selection; the
benchmark runs at the explicitly supplied shape, not across the entire profile. Workspace limits
builder tactic workspace and does not cap total inference memory. A timing cache stores tactic
measurements, whereas an engine is the compiled deployment artifact.

The FP32 baseline uses `--noTF32`. FP16 builds retain TensorRT's default TF32 policy for any FP32
work, including the excluded detection head; hardware may not support TF32. The report therefore
compares documented deployment modes, rather than attributing all speedup solely to FP16. Every
run uses the same warmup/duration settings, transfers enabled, one inference stream, and a separate
profiling pass. Compare at the same shape. `trtexec` random inputs measure engine execution and
exclude image preprocessing, detector decode, and NMS; use lesson 07 for backend accuracy checks.

AutoCast excludes `/model.22/.*` to keep YOLOv8's numerically sensitive detection head in FP32 and
keeps callable I/O as float32. The conversion gate uses the saved image at `[1,3,640,640]` for static
and a repeated batch `[2,3,640,640]` for dynamic. Coordinate channels use `rtol=0.05, atol=0.05`;
score channels use `rtol=0.01, atol=0.02`. These are deliberately permissive conversion smoke limits,
not production accuracy requirements: at a coordinate of 500 the coordinate gate permits an error
of 25.05 pixels. The recorded conversion run had maximum coordinate differences around 3 pixels
and score differences below 0.005. Review distributions and downstream effects rather than relaxing
tolerances until a result passes. A passing single-image raw-output gate does not establish mAP or
accuracy across dynamic shapes; later lessons extend that validation.
