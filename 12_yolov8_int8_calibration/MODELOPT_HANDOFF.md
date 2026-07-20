# Precision Recovery Step 06: TensorRT 10 Native FP16 Q/DQ Handoff

This handoff replaces the earlier ModelOpt plans. It defines the complete execution contract for
Lesson 12 precision-recovery Step 06. Read the repository-root `AGENTS.md`, this file, and
`precision_recovery/05_modelopt_ptq/README.md` before making changes.

## Objective

Build a complete, version-matched TensorRT 10.14 evidence chain for the native ModelOpt
INT8-plus-FP16 Q/DQ graph:

1. build new TensorRT 10 FP32 and FP16 reference engines;
2. build an optimized strongly-typed TensorRT 10 INT8+FP16 candidate;
3. inspect and record the actual INT8, FP16, FP32, and reformat layer structure;
4. run PyTorch, TensorRT 10 FP32, TensorRT 10 FP16, and TensorRT 10 INT8 over the unchanged complete
   5,000-image COCO val2017 split;
5. apply the original four accuracy thresholds without modification;
6. run matched FP32/FP16/INT8 `trtexec` performance measurements;
7. document the quality and performance decision with complete artifact identities.

This is PTQ-only. Do not perform QAT, training, fine-tuning, pseudo-label training, threshold
relaxation, or repeated validation-label tuning.

## Step Directory

Create the ordered recovery module:

```text
12_yolov8_int8_calibration/precision_recovery/06_trt10_native_fp16_qdq/
```

Recommended committed files:

```text
README.md
build_trt10_evidence.py
benchmark_trt10_evidence.py
inspect_trt10_layers.py
test_build_trt10_evidence.py
test_benchmark_trt10_evidence.py
test_inspect_trt10_layers.py
```

Keep the public APIs narrow. Build orchestration, Engine Inspector analysis, benchmarking, and
reporting should be separate responsibilities. Reuse the existing Lesson 12 evaluator rather than
copying metric or postprocessing logic.

## Required Container

Use the persistent ModelOpt/TensorRT 10 container for every CUDA-, TensorRT-, PyTorch-, ONNX-, and
GPU-dependent Step 06 operation:

```bash
docker start learn-tensorrt-modelopt
docker exec -it learn-tensorrt-modelopt bash
cd /workspace/Learn-TensorRT
```

The repository is bind-mounted at `/workspace/Learn-TensorRT`.

Recorded environment:

- image: `nvcr.io/nvidia/pytorch:25.11-py3`
- GPU: NVIDIA GeForce RTX 2060, compute capability 7.5, 6 GiB
- driver: `580.159.04`
- CUDA runtime: 13.0
- PyTorch: `2.10.0a0+b558c986e8.nv25.11`
- ModelOpt: `0.37.0`
- TensorRT: `10.14.1.48`
- ONNX: `1.18.0`
- Ultralytics: `8.4.22`
- OpenCV: `4.13.0`
- NumPy: `2.1.0`

PyTorch warns that the RTX 2060 is compute capability 7.5 and recommends CUDA 12.6/12.8 builds.
Preserve the warning in reproducibility notes. Do not claim broader compatibility beyond the checks
that were actually run.

Do not use `trt_dev` to build or evaluate Step 06 engines. It remains historical TensorRT 8.6.1
evidence only.

## Starting Git State

Relevant completed commits:

```text
192ca09 Add ModelOpt QDQ INT8 accuracy recovery
7f7c216 Record ModelOpt QDQ throughput
8be934b Record FP16 QDQ parser limitation
```

Inspect `git status --short` before editing. Preserve unrelated user changes. Do not create or switch
branches unless explicitly requested.

## Immutable Source Artifacts

### PyTorch weights

```text
assets/yolov8n.pt
SHA-256: f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36
```

### Canonical FP32 ONNX reference graph

```text
05_torch_to_onnx/outputs/yolov8n.onnx
SHA-256: 738879ea1d605f6bdce1454120cf02b89cc5fdbbc233dcf2d5a4120762b96260
```

Use this graph to build the TensorRT 10 FP32 and FP16 references. Do not regenerate it silently. If
its hash differs, stop and explain the identity change before continuing.

### Native FP16-high-precision ModelOpt Q/DQ graph

```text
12_yolov8_int8_calibration/outputs/precision_recovery/05_modelopt_ptq/
yolov8n_modelopt_int8_max_hp_fp16_train3000.onnx
SHA-256: 3fad6a3dba71e4026c7e8036a413fbc027dd310cf15c91e72ccdda70e677dc90
```

Metadata:

```text
12_yolov8_int8_calibration/outputs/precision_recovery/05_modelopt_ptq/
yolov8n_modelopt_int8_max_hp_fp16_train3000.onnx.json
```

Current metadata SHA-256:

```text
ee33a326da8898189a8a9a95a62d2c7ae8fe46dba6e1df518ecff10e2761bc38
```

The metadata file may be regenerated only when the ONNX-inspection schema changes. The ONNX hash is
the authoritative model identity.

The Q/DQ graph contract is:

```text
images  FLOAT [1, 3, 640, 640]
  -> Cast to FLOAT16
  -> native FLOAT16 high-precision tensors plus INT8 Q/DQ
  -> Cast to FLOAT
output0 FLOAT [1, 84, 8400]
```

Verified graph evidence:

- 131 `QuantizeLinear` nodes;
- 131 `DequantizeLinear` nodes;
- 2 FP32/FP16 boundary `Cast` nodes;
- 135 FLOAT16 tensor constants;
- all 262 Q/DQ scale uses audited;
- zero non-positive scales;
- zero positive FP16 subnormal scales;
- ONNX checker passed.

Do not recalibrate or re-export this graph by default. Step 06 studies the already-declared native
FP16 Q/DQ candidate. Re-export only if the source artifact is missing or its identity cannot be
verified, and then reproduce the exact Step 05 command and record the new identity.

## Prior Evidence And Why Step 06 Exists

The FP32-high-precision Q/DQ graph built successfully in TensorRT 8.6 and passed the quality gate,
but matched throughput was worse than FP16:

| Engine | Throughput (qps) | GPU compute mean (ms) | Quality gate |
| --- | ---: | ---: | --- |
| TensorRT 8.6 FP16 reference | 650.348 | 1.523 | PASS |
| TensorRT 8.6 FP32-high Q/DQ INT8+FP16 | 618.236 | 1.602 | PASS |

Engine Inspector showed 67 reformats among 171 engine layers and a remaining FP32 detection-head
path. Step 05 therefore exported native FP16 Q/DQ high-precision tensors.

TensorRT 8.6 rejected the native Half Q/DQ graph at the first per-channel weight `QuantizeLinear`
with:

```text
Assertion failed: scaleAllPositive && "Scale coefficients must all be positive"
```

The scale audit proved the scales were valid. TensorRT 10.14 then parsed and built the same graph
successfully with:

```bash
trtexec \
  --onnx=12_yolov8_int8_calibration/outputs/precision_recovery/05_modelopt_ptq/\
yolov8n_modelopt_int8_max_hp_fp16_train3000.onnx \
  --stronglyTyped \
  --builderOptimizationLevel=0 \
  --skipInference
```

The compatibility build completed in approximately 16.98 seconds. It did not save an engine and is
not performance or accuracy evidence. Step 06 promotes this graph to a complete optimized TensorRT
10 experiment.

## Fixed Dataset And Accuracy Contract

Use the coverage-aware manifest:

```text
12_yolov8_int8_calibration/outputs/precision_recovery/02_calibration_coverage/
dataset_manifest.json
```

Dataset identity:

```text
coco2017-train3000-coverage-v2-seed42-calibration-val5000-human-labels-v1
```

Manifest SHA-256 recorded by the existing evaluation evidence:

```text
988af7ecdd34e3ccbb319d5a6e5611434f347502d57310877ae9ddd6e6de6389
```

The validation contract is immutable:

- validation images: all 5,000 COCO val2017 images declared by the manifest;
- preprocessing: letterbox 114, BGR-to-RGB, FP32 divide by 255, NCHW;
- confidence threshold: `0.001`;
- NMS IoU threshold: `0.7`;
- maximum detections: `300`;
- metric: `course-coco-like-101point-v2-no-crowd-no-area-ranges`;
- maximum mAP50-95 drop versus PyTorch: `0.02`;
- maximum mAP50 drop: `0.02`;
- maximum precision drop: `0.03`;
- maximum recall drop: `0.03`.

Do not use validation images for calibration, candidate selection, layer selection, or threshold
tuning. Do not claim this metric is official `pycocotools` COCO AP.

## Matched Reference Requirement

Build new TensorRT 10 FP32 and FP16 engines. Do not deserialize, reuse, or identity-link the older
TensorRT 8.6 engines.

Mathematically, the quality gate is evaluated against PyTorch. The TensorRT 10 references are still
required because they provide:

- a matched parser/runtime FP32 boundary check;
- a matched FP16 deployment baseline;
- TensorRT FP32 raw-output drift evidence;
- a defensible FP16-versus-INT8 performance comparison;
- a complete four-backend report using one software identity.

Do not implement a shortcut that reuses the old TensorRT 8.6 reference report.

## Planned Build Artifacts

Put generated artifacts under the ignored directory:

```text
12_yolov8_int8_calibration/outputs/precision_recovery/06_trt10_native_fp16_qdq/
```

Recommended layout:

```text
references/
  yolov8n_trt10_fp32.engine
  yolov8n_trt10_fp32.build.log
  yolov8n_trt10_fp32.layers.json
  yolov8n_trt10_fp16.engine
  yolov8n_trt10_fp16.build.log
  yolov8n_trt10_fp16.layers.json
  reference_builds.json
candidate/
  yolov8n_modelopt_hp_fp16_trt10.engine
  yolov8n_modelopt_hp_fp16_trt10.build.log
  yolov8n_modelopt_hp_fp16_trt10.layers.json
  yolov8n_modelopt_hp_fp16_trt10.engine.json
evaluation/
  precision_evaluation.json
  precision_evaluation.md
  run.log
performance/
  fp32_times.json
  fp16_times.json
  int8_times.json
  fp32_trtexec.log
  fp16_trtexec.log
  int8_trtexec.log
  performance.json
```

Use separate timing caches for the canonical FP32/FP16 graph and the native Q/DQ graph. Do not reuse
TensorRT 8.6 timing caches.

## Build Contracts

### TensorRT 10 FP32 reference

Build from the canonical FP32 ONNX. Prefer strongly typed construction so the FLOAT graph remains
FP32:

```bash
trtexec \
  --onnx=05_torch_to_onnx/outputs/yolov8n.onnx \
  --saveEngine=<step06>/references/yolov8n_trt10_fp32.engine \
  --stronglyTyped \
  --builderOptimizationLevel=3 \
  --skipInference \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --exportLayerInfo=<step06>/references/yolov8n_trt10_fp32.layers.json \
  --timingCacheFile=<step06>/references/trt10_reference.timing.cache
```

If TensorRT rejects strongly typed construction for the canonical graph, stop and record the exact
reason before changing flags. Do not silently fall back to a mixed-precision FP32 reference.

### TensorRT 10 FP16 reference

Build from the same canonical FP32 ONNX with FP16 enabled:

```bash
trtexec \
  --onnx=05_torch_to_onnx/outputs/yolov8n.onnx \
  --saveEngine=<step06>/references/yolov8n_trt10_fp16.engine \
  --fp16 \
  --builderOptimizationLevel=3 \
  --skipInference \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --exportLayerInfo=<step06>/references/yolov8n_trt10_fp16.layers.json \
  --timingCacheFile=<step06>/references/trt10_reference.timing.cache
```

Do not use `--stronglyTyped` for this FP16 reference because the source ONNX tensors are FLOAT and
strong typing would preserve FP32.

### Native ModelOpt INT8+FP16 candidate

Build the immutable Half Q/DQ ONNX with strongly typed construction:

```bash
trtexec \
  --onnx=12_yolov8_int8_calibration/outputs/precision_recovery/05_modelopt_ptq/\
yolov8n_modelopt_int8_max_hp_fp16_train3000.onnx \
  --saveEngine=<step06>/candidate/yolov8n_modelopt_hp_fp16_trt10.engine \
  --stronglyTyped \
  --builderOptimizationLevel=3 \
  --skipInference \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  --exportLayerInfo=<step06>/candidate/yolov8n_modelopt_hp_fp16_trt10.layers.json \
  --timingCacheFile=<step06>/candidate/trt10_qdq.timing.cache
```

Do not add `--fp16` or `--int8` to the strongly-typed candidate command. The ONNX Q/DQ and tensor
types define the precision contract.

Every build wrapper must record:

- the exact argument vector;
- TensorRT version;
- input ONNX path and SHA-256;
- engine path, byte size, and SHA-256;
- timing-cache path and SHA-256;
- complete build log and SHA-256;
- layer-info path and SHA-256;
- engine input/output names, data types, and static shapes;
- build duration when available from the log.

Required engine boundary for every backend:

```text
images  FLOAT [1, 3, 640, 640]
output0 FLOAT [1, 84, 8400]
```

Stop if any engine has a different boundary contract.

## Layer And Precision Inspection

Do not report raw keyword counts as compute-layer counts. Parse the Engine Inspector JSON and
separate:

- compute layers: convolution, pointwise, elementwise, pooling, resize, softmax, and other kernels;
- infrastructure layers: reformat, shuffle, no-op, constant, shape, and copies;
- INT8-weight convolutions grouped by output type;
- pure FP16 compute layers;
- pure FP32 compute layers;
- FP32 external boundary conversions;
- total reformat count and Q/DQ-origin reformat count.

Compare the native TensorRT 10 candidate against the Step 05 TensorRT 8.6 FP32-high Q/DQ evidence:

- Step 05 engine: 171 total layers, including 67 reformats;
- Step 05 compute-layer output categories: 49 INT8, 32 FP16, 12 FP32;
- only five non-INT8 pure FP32 compute layers were identified, all near final YOLO decode/output.

The Step 06 report must state whether native Half Q/DQ materially reduces FP32 compute and format
conversion. Do not infer performance improvement from layer counts alone.

## Pre-Gate Unlabeled Checks

Before using validation labels:

1. run all three TensorRT 10 engines on the same small set of calibration images;
2. verify output shape, dtype, finiteness, and deterministic repeatability;
3. compare raw outputs against PyTorch or TensorRT 10 FP32;
4. record maximum, mean, and P99 absolute drift for FP16 and INT8;
5. confirm that the candidate is not producing NaNs, infinities, collapsed class scores, or invalid
   box ranges.

Use calibration images only for this sensitivity check. Do not use the validation labels to decide
whether to proceed. A runtime failure, non-finite output, wrong shape, or clearly corrupted output is
a build failure and should stop the formal gate.

## TensorRT 10 Evaluation Plan

The existing `compare_engines.py` and Lesson 09 helper already contain TensorRT 10 tensor-name API,
`set_tensor_address`, and `execute_async_v3` compatibility branches. Preserve TensorRT 8.6 fallback
code.

Run a full four-backend evaluation in `learn-tensorrt-modelopt`:

```text
PyTorch in the TensorRT 10 container
TensorRT 10 FP32
TensorRT 10 FP16
TensorRT 10 native Q/DQ INT8+FP16
```

Do not reuse the TensorRT 8.6 reference report. The full run must create a new TensorRT 10 report
with current package versions and engine hashes.

Planned command shape:

```bash
docker exec learn-tensorrt-modelopt bash -lc '
  cd /workspace/Learn-TensorRT/12_yolov8_int8_calibration &&
  python3 compare_engines.py \
    --manifest outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json \
    --weights ../assets/yolov8n.pt \
    --fp32-engine outputs/precision_recovery/06_trt10_native_fp16_qdq/references/\
yolov8n_trt10_fp32.engine \
    --fp16-engine outputs/precision_recovery/06_trt10_native_fp16_qdq/references/\
yolov8n_trt10_fp16.engine \
    --int8-engine outputs/precision_recovery/06_trt10_native_fp16_qdq/candidate/\
yolov8n_modelopt_hp_fp16_trt10.engine \
    --output-dir outputs/precision_recovery/06_trt10_native_fp16_qdq/evaluation
'
```

The report must preserve:

- all four backend metrics;
- deltas versus the newly measured PyTorch backend;
- engine and weights hashes;
- complete software identity;
- TensorRT FP16 and INT8 raw drift versus TensorRT FP32;
- the unchanged thresholds;
- release-gate process exit status;
- changed/high-drift examples collected by the existing evaluator.

Do not claim a gate pass unless the complete 5,000-image run finishes and the JSON report says all
four original thresholds passed.

## Performance Plan

After the accuracy gate, collect matched `trtexec` evidence for all three TensorRT 10 engines:

- warmup: 500 ms;
- measured iterations: 120;
- duration: 0;
- data transfers: enabled;
- one inference stream;
- input shape: static model shape;
- export raw per-inference timing JSON;
- use the wall-time throughput printed by `trtexec`;
- do not invert per-inference latency to estimate throughput.

Record for FP32, FP16, and INT8:

- throughput qps;
- end-to-end latency mean, P50, P90, and P99;
- GPU compute mean, P50, P90, and P99;
- H2D and D2H means;
- engine hash;
- exact command and log.

Run the engines back-to-back in the same container session. Note GPU name, driver, clocks/power state
when available, and any thermal or contention limitation.

Quality controls release eligibility. Performance never overrides a failed quality gate.

Decision rules:

- gate fails: record the candidate as failed; retain the established FP16 deployment decision;
- gate passes but INT8 is not faster than matched FP16: keep FP16 for deployment and retain INT8 as
  quality evidence;
- gate passes and INT8 has a reproducible performance benefit: select the native Q/DQ candidate,
  preferring the simplest passing configuration with no unnecessary FP32 compute fallback.

## Candidate Discipline

The native Half Q/DQ graph is the only predeclared Step 06 formal INT8 candidate. Run the complete
5,000-image gate once after all build and unlabeled checks pass.

If it fails quality:

- preserve the report and artifact identities;
- do not loosen thresholds;
- do not repeatedly change layers based on validation labels;
- do not automatically start QAT;
- stop and propose a new predeclared recovery experiment based on calibration-output sensitivity or
  documented TensorRT limitations.

If a reference backend fails the gate, diagnose the reference/runtime alignment before interpreting
the INT8 result.

## Detailed Execution Sequence

### Phase 0: Preflight

1. Confirm the worktree state.
2. Start `learn-tensorrt-modelopt`.
3. Verify GPU visibility and package versions.
4. Verify the four immutable hashes: weights, FP32 ONNX, Half Q/DQ ONNX, manifest.
5. Confirm output directories are ignored.
6. Confirm no stale TensorRT 10 build or evaluation process is running.

### Phase 1: Step 06 implementation

1. Create `precision_recovery/06_trt10_native_fp16_qdq/`.
2. Add focused build, inspection, benchmark, and metadata modules.
3. Add CPU tests for command construction, identity validation, Inspector classification, timing
   summary, and failure handling.
4. Add a concise Step 06 README with commands and decision rules.

### Phase 2: Reference builds

1. Build TensorRT 10 FP32 from the canonical ONNX.
2. Validate FP32 boundary I/O.
3. Build TensorRT 10 FP16 from the same ONNX.
4. Validate FP16 boundary I/O.
5. Record hashes, logs, timing caches, and Inspector JSON.

### Phase 3: Candidate build

1. Build the Half Q/DQ graph with strongly typed TensorRT 10 and optimization level 3.
2. Validate the FP32 external I/O contract.
3. Record the exact artifact identity.
4. Stop on any parser warning that changes Q/DQ semantics or any build failure.

### Phase 4: Layer audit and unlabeled sensitivity

1. Generate classified layer summaries for all three engines.
2. Confirm the candidate contains INT8-weight compute and FP16 high-precision compute.
3. Quantify pure FP32 compute and reformats.
4. Run calibration-image raw-output checks.
5. Stop before the gate if outputs are invalid or corrupted.

### Phase 5: Complete quality gate

1. Run PyTorch plus all three TensorRT 10 engines on 5,000 images.
2. Compute the existing course metrics.
3. Preserve the process exit code and complete JSON/Markdown report.
4. Do not rerun with changed thresholds or postprocessing.

### Phase 6: Matched performance

1. Run the 120-iteration benchmark for FP32, FP16, and INT8.
2. Generate the identity-linked performance report.
3. Compare INT8 directly with matched FP16.

### Phase 7: Documentation and verification

1. Update `precision_recovery/README.md` with the Step 06 result.
2. Update the Lesson 12 README release decision.
3. If the experiment fails before gate, record the detailed failure in the Step 06 README.
4. Run all Step 06 tests and the relevant Lesson 12 tests.
5. Run `git diff --check`.
6. State exactly which builds, tests, evaluation, and benchmarks were actually run.

## Failure Handling

Record failures at the phase where they occur. A failure record must include:

- candidate/configuration ID;
- source artifact hashes;
- exact command;
- container and package versions;
- complete error text or log path;
- whether an engine was created;
- whether validation labels were consulted;
- whether the full gate was run;
- the next technically justified option without silently starting it.

Do not overwrite successful or failed Step 05 evidence.

## Repository And Artifact Rules

- Generated engines, timing caches, ONNX copies, raw tensor dumps, logs, layer JSON, timing JSON, and
  local reports stay under ignored `outputs/` directories.
- Commit source scripts, focused tests, concise README documentation, and intentionally curated
  small metadata only.
- Do not commit serialized TensorRT engines unless explicitly requested.
- Do not commit large raw Inspector or timing captures.
- Do not add `_practice` directories, TODO-only copies, or solution branches.
- Preserve compatibility with the pinned containers; do not upgrade packages.

## Acceptance Criteria

Step 06 is complete only when:

- new TensorRT 10 FP32 and FP16 references are built and identity-recorded;
- the optimized strongly-typed native INT8+FP16 engine is built and identity-recorded;
- all three engines expose the exact FP32 static I/O contract;
- Inspector evidence classifies actual compute and infrastructure precision;
- unlabeled raw-output checks pass;
- the complete unchanged 5,000-image four-backend evaluation finishes;
- the original four thresholds determine the gate result;
- matched FP32/FP16/INT8 performance is measured with 120 samples;
- the deployment decision follows quality first, then matched FP16-versus-INT8 performance;
- focused tests and `git diff --check` pass;
- documentation records both successful and unverified items precisely.

## First Commands For The Next Window

```bash
cd /home/parker/Projects/Learn-TensorRT
cat AGENTS.md
cat 12_yolov8_int8_calibration/MODELOPT_HANDOFF.md
git -c safe.directory=/home/parker/Projects/Learn-TensorRT status --short

docker start learn-tensorrt-modelopt
docker exec learn-tensorrt-modelopt bash -lc '
  cd /workspace/Learn-TensorRT &&
  python3 -c "import modelopt, modelopt.torch.quantization, tensorrt, torch, onnx, cv2" &&
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader &&
  sha256sum \
    assets/yolov8n.pt \
    05_torch_to_onnx/outputs/yolov8n.onnx \
    12_yolov8_int8_calibration/outputs/precision_recovery/05_modelopt_ptq/\
yolov8n_modelopt_int8_max_hp_fp16_train3000.onnx \
    12_yolov8_int8_calibration/outputs/precision_recovery/02_calibration_coverage/\
dataset_manifest.json
'
```

After preflight, begin by scaffolding Step 06 and implementing the versioned build/metadata wrapper.
