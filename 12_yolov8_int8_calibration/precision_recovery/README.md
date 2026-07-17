# INT8 Precision Recovery Log

This directory records ordered, one-variable-at-a-time experiments for recovering the Lesson 12
INT8 accuracy regression. Generated JSON evidence stays under the lesson's ignored `outputs/`
directory. The fixed COCO validation split and predeclared accuracy thresholds must not change
during these experiments.

## Outcome Summary

The TensorRT 8.6.1 legacy-calibrator recovery sequence is complete. MinMax calibration on the
coverage-aware 3,000-image split recovered most of the original Entropy regression. Explicit FP16
profiles for final box outputs, final box-and-class outputs, and the complete detection head were
then evaluated under the unchanged 5,000-image gate.

The strongest accuracy candidate used MinMax with the complete detection head in FP16. It achieved
`0.3463` mAP50-95 and `0.4897` mAP50, retained approximately `20.7%` higher matched throughput than
FP16, and still failed because its mAP50 drop was `0.02057` against the allowed `0.02`. FP16 remains
the release candidate. Explicit Q/DQ or QAT is intentionally deferred to a separate version-pinned
environment; no ModelOpt package was installed here.

## 01 - Preprocessing Parity

Status: **PASS** on 2026-07-16.

The verifier independently calls the production calibration and evaluation preprocessing paths. It
requires exact byte equality after letterbox resize, padding with 114, BGR-to-RGB conversion,
FP32 division by 255, HWC-to-CHW conversion, and contiguous layout. Synthetic tests include odd
dimensions and extreme aspect ratios; the manifest verifier checks the complete hashed calibration
split by default.

Run inside the pinned TensorRT development container from the repository root:

```bash
python3 -m unittest discover \
  -s 12_yolov8_int8_calibration/precision_recovery/01_preprocessing_parity -v
python3 \
  12_yolov8_int8_calibration/precision_recovery/01_preprocessing_parity/verify_preprocessing_parity.py
```

Evidence: `12_yolov8_int8_calibration/outputs/precision_recovery/01_preprocessing_parity.json`.

Recorded result: all 1,000 calibration images produced byte-identical `(1, 3, 640, 640)` FP32
contiguous tensors through both production paths. Zero images failed; 4,915,200,000 tensor bytes
were compared. This rules out preprocessing-path mismatch as the cause of the current INT8 accuracy
drop. The generated JSON also records the dataset manifest and implementation SHA-256 identities.

## 02 - Versioned Calibration Coverage

Status: **FAIL** on 2026-07-17.

The experiment retained the canonical 1,000 images and selected 2,000 additions from a deterministic
4,000-image train2017 candidate pool. Selection first balances category participation, then applies
farthest-point coverage sampling over object counts, small/medium/large ratios, image and box aspect
statistics, luminance, contrast, dark/highlight fractions, saturation, and edge density.

The candidate manifest ID is
`coco2017-train3000-coverage-v2-seed42-calibration-val5000-human-labels-v1`. The fixed 5,000-image
validation split and every declared gate remained unchanged. All 3,000 calibration images also
passed the byte-identical preprocessing check.

| Calibration | mAP50-95 | mAP50 | Precision | Recall | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Entropy, 1,000 images | 0.3179 | 0.4560 | 0.0429 | 0.7858 | FAIL |
| Entropy, 3,000 images | 0.3247 | 0.4651 | 0.0429 | 0.7940 | FAIL |

The larger split recovered `0.0068` mAP50-95, `0.0091` mAP50, and `0.0083` recall, but remained
below the predeclared mAP gates. Per the experiment decision, the calibration set was not expanded
to 5,000 images before testing a different calibration algorithm.

Implementation:
`precision_recovery/02_calibration_coverage/prepare_coverage_calibration.py`.

Evidence:

- `outputs/precision_recovery/02_calibration_coverage/coverage_report.json`
- `outputs/precision_recovery/02_calibration_coverage/preprocessing_parity.json`
- `outputs/precision_recovery/02_calibration_coverage/evaluation/precision_evaluation.json`

## 03 - Entropy Versus MinMax

Status: **FAIL** on 2026-07-17.

This controlled experiment changes only the TensorRT PTQ calibration algorithm. It reuses the exact
3,000-image Step 02 manifest, ONNX model, input shape, preprocessing, workspace, FP16 fallback,
5,000-image validation split, postprocessing settings, and accuracy gates.

Calibration algorithm is part of the cache identity and cache metadata. Entropy and MinMax use
different cache, engine, and report paths so TensorRT cannot silently reuse the wrong calibration
table.

Run from `12_yolov8_int8_calibration`:

```bash
python3 build_int8_engine.py \
  --calibrator minmax \
  --manifest outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json \
  --cache outputs/precision_recovery/03_entropy_vs_minmax/yolov8n_minmax_train3000.cache \
  --output outputs/precision_recovery/03_entropy_vs_minmax/yolov8n_minmax_train3000.engine \
  --enable-fp16

python3 compare_engines.py \
  --manifest outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json \
  --int8-engine \
    outputs/precision_recovery/03_entropy_vs_minmax/yolov8n_minmax_train3000.engine \
  --output-dir outputs/precision_recovery/03_entropy_vs_minmax/evaluation
```

Recorded result:

| Calibration | mAP50-95 | mAP50 | Precision | Recall | Mean latency (ms) | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Entropy, 3,000 images | 0.3247 | 0.4651 | 0.0429 | 0.7940 | 4.110 | FAIL |
| MinMax, 3,000 images | 0.3444 | 0.4892 | 0.0429 | 0.7936 | 4.060 | FAIL |

MinMax recovered `0.0197` mAP50-95 and `0.0241` mAP50 relative to the controlled 3,000-image
Entropy candidate. Its mAP50-95 drop from PyTorch was `0.01874`, so that metric passed the allowed
`0.02` drop. Precision and recall also passed. The only failed metric was mAP50: its drop was
`0.02098`, exceeding the predeclared `0.02` limit by approximately `0.00098`.

The release gate therefore remains failed. The threshold must not be relaxed after observing this
near miss. MinMax is the strongest PTQ candidate so far and should be the baseline for the next
layer-sensitivity experiment. Raw output drift increased despite the task metrics improving, which
also confirms that raw tensor drift is diagnostic evidence rather than the release criterion.

Evidence:
`outputs/precision_recovery/03_entropy_vs_minmax/evaluation/precision_evaluation.json`.

## 04 - Detection-Head Layer Sensitivity

Status: **FAIL** on 2026-07-17.

The first mixed-precision candidate keeps the 3,000-image MinMax calibration experiment unchanged
and constrains only the three final box-regression convolutions to FP16. These P3/P4/P5 layers emit
the 64 DFL regression logits immediately before reshape, concatenation, and DFL decoding:

```text
/model.22/cv2.0/cv2.0.2/Conv
/model.22/cv2.1/cv2.1.2/Conv
/model.22/cv2.2/cv2.2.2/Conv
```

The builder uses `OBEY_PRECISION_CONSTRAINTS`, sets both computation and output type to FP16, and
fails if any expected layer name is absent or is no longer a convolution. Classification outputs,
the rest of the detection head, neck, and backbone remain eligible for INT8.

Run from `12_yolov8_int8_calibration`:

```bash
python3 build_int8_engine.py \
  --calibrator minmax \
  --precision-profile box_outputs_fp16 \
  --manifest outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json \
  --cache outputs/precision_recovery/03_entropy_vs_minmax/yolov8n_minmax_train3000.cache \
  --output \
    outputs/precision_recovery/04_layer_sensitivity/yolov8n_minmax_box_outputs_fp16.engine \
  --enable-fp16

python3 compare_engines.py \
  --manifest outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json \
  --int8-engine \
    outputs/precision_recovery/04_layer_sensitivity/yolov8n_minmax_box_outputs_fp16.engine \
  --output-dir outputs/precision_recovery/04_layer_sensitivity/box_outputs_fp16_evaluation
```

Engine Inspector confirmed that all three target convolutions use FP16 inputs, weights, tactics,
and outputs. The surrounding detection-head layers remained eligible for INT8; explicit reformat
layers convert the upstream INT8 activations at the constrained boundaries.

Recorded result:

| Candidate | mAP50-95 | mAP50 | Precision | Recall | Mean latency (ms) | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MinMax, no explicit constraints | 0.3444 | 0.4892 | 0.0429 | 0.7936 | 4.060 | FAIL |
| MinMax, box outputs FP16 | 0.3452 | 0.4893 | 0.0429 | 0.7936 | 4.076 | FAIL |

Protecting the three final regression convolutions recovered `0.00084` mAP50-95 but only
`0.00008` mAP50. The mAP50 drop from PyTorch remained `0.02090`, exceeding the allowed `0.02`
drop by approximately `0.00090`. The candidate therefore failed the unchanged gate. Mean wrapper
latency increased by approximately `0.017 ms`; this is diagnostic timing rather than the matched
`trtexec` performance result.

This result indicates that the remaining mAP50 regression is not primarily caused by quantization
of the three final box-regression convolutions. No classification layers or broader detection-head
regions were constrained as part of this candidate.

Evidence:

- `outputs/precision_recovery/04_layer_sensitivity/`
  `yolov8n_minmax_box_outputs_fp16.engine.json`
- `outputs/precision_recovery/04_layer_sensitivity/`
  `yolov8n_minmax_box_outputs_fp16.engine.layers.json`
- `outputs/precision_recovery/04_layer_sensitivity/box_outputs_fp16_evaluation/`
  `precision_evaluation.json`

### Faster Candidate Iteration

Full four-backend evaluation is required when the reference model, FP32/FP16 engines, validation
manifest, software environment, preprocessing, postprocessing, metric implementation, or gate
changes. It is not required for every mixed-precision candidate when those identities remain fixed.

`compare_engines.py --reference-report` validates all reference artifact hashes, software versions,
dataset identity, input shape, evaluation settings, metric implementation, and thresholds before
reusing the recorded PyTorch, TensorRT FP32, and TensorRT FP16 metrics. Only the new candidate engine
runs over val2017. Candidate-only mode does not recompute raw FP32 tensor drift or changed-example
diagnostics; a full run remains available when those diagnostics are needed.

TensorRT engine builds also accept `--timing-cache`. A compatible persistent timing cache avoids
remeasuring tactics that were already profiled during earlier candidates. TensorRT rejects an
incompatible cache rather than silently accepting mismatched timing data.

### Combined Box And Classification Outputs

Status: **FAIL** on 2026-07-17.

The next controlled candidate combines the three previously tested box-regression output
convolutions with the three final classification output convolutions. Only these six 1x1 output
layers are explicitly FP16; earlier detection-head layers remain eligible for INT8.

```text
/model.22/cv2.0/cv2.0.2/Conv
/model.22/cv2.1/cv2.1.2/Conv
/model.22/cv2.2/cv2.2.2/Conv
/model.22/cv3.0/cv3.0.2/Conv
/model.22/cv3.1/cv3.1.2/Conv
/model.22/cv3.2/cv3.2.2/Conv
```

Run from `12_yolov8_int8_calibration`:

```bash
python3 build_int8_engine.py \
  --calibrator minmax \
  --precision-profile box_and_class_outputs_fp16 \
  --manifest outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json \
  --cache outputs/precision_recovery/03_entropy_vs_minmax/yolov8n_minmax_train3000.cache \
  --timing-cache outputs/precision_recovery/04_layer_sensitivity/tactics.cache \
  --output \
    outputs/precision_recovery/04_layer_sensitivity/yolov8n_minmax_box_and_class_outputs_fp16.engine \
  --enable-fp16

python3 compare_engines.py \
  --reference-report \
    outputs/precision_recovery/03_entropy_vs_minmax/evaluation/precision_evaluation.json \
  --manifest outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json \
  --int8-engine \
    outputs/precision_recovery/04_layer_sensitivity/yolov8n_minmax_box_and_class_outputs_fp16.engine \
  --output-dir \
    outputs/precision_recovery/04_layer_sensitivity/box_and_class_outputs_fp16_evaluation
```

Engine Inspector confirmed FP16 inputs, weights, tactics, and outputs for all six constrained
convolutions. The persistent timing cache was written successfully for reuse by later compatible
builds. Candidate-only evaluation validated and reused the Step 03 PyTorch, TensorRT FP32, and
TensorRT FP16 identities, then ran only the combined mixed-precision engine over val2017.

Recorded result:

| Candidate | mAP50-95 | mAP50 | Precision | Recall | Mean latency (ms) | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MinMax, no explicit constraints | 0.3444 | 0.4892 | 0.0429 | 0.7936 | 4.060 | FAIL |
| MinMax, box outputs FP16 | 0.3452 | 0.4893 | 0.0429 | 0.7936 | 4.076 | FAIL |
| MinMax, box and class outputs FP16 | 0.3447 | 0.4892 | 0.0429 | 0.7936 | 4.062 | FAIL |

Relative to the unconstrained MinMax candidate, protecting all six output convolutions changed
mAP50-95 by `+0.00037` and mAP50 by `-0.00007`. Its mAP50 drop from PyTorch was `0.02105`, so the
candidate remained outside the allowed `0.02` drop. Classification-output fallback did not recover
the remaining gate failure, and combining it with the box-output fallback was weaker than the
box-only result.

The candidate-only path completed the 5,000-image inference and single-backend metric calculation
without rerunning the three unchanged reference backends. Raw FP32 drift and changed-example
diagnostics were intentionally omitted in this fast mode; the unchanged task-level gate remained
authoritative.

This result does not support expanding FP16 protection further through the terminal output layers.
A broader detection-head experiment would change substantially more computation and should be
treated as a separate decision. Explicit Q/DQ placement or QAT is now more informative than adding
more output-layer constraints one region at a time.

Evidence:

- `outputs/precision_recovery/04_layer_sensitivity/`
  `yolov8n_minmax_box_and_class_outputs_fp16.engine.json`
- `outputs/precision_recovery/04_layer_sensitivity/`
  `yolov8n_minmax_box_and_class_outputs_fp16.engine.layers.json`
- `outputs/precision_recovery/04_layer_sensitivity/box_and_class_outputs_fp16_evaluation/`
  `precision_evaluation.json`

### Complete Detection Head FP16

Status: **FAIL** on 2026-07-17.

As the final legacy-PTQ recovery candidate, the complete `model.22.cv2.*` regression branches and
`model.22.cv3.*` classification branches are constrained to FP16. The profile includes all 18
convolutions, 12 Sigmoid activations, and 12 elementwise SiLU multiplications in the P3/P4/P5 head.
Shape operations, concatenation, DFL/decode operations, neck, and backbone are not added to the
profile. TensorRT already keeps unsupported or unscaled DFL operations outside INT8.

The builder validates the expected 42-layer detection-head structure before applying the profile.
Any export change that alters the expected layer types or counts stops the build instead of silently
using a partial fallback.

```bash
python3 build_int8_engine.py \
  --calibrator minmax \
  --precision-profile detection_head_fp16 \
  --manifest outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json \
  --cache outputs/precision_recovery/03_entropy_vs_minmax/yolov8n_minmax_train3000.cache \
  --timing-cache outputs/precision_recovery/04_layer_sensitivity/tactics.cache \
  --output \
    outputs/precision_recovery/04_layer_sensitivity/yolov8n_minmax_detection_head_fp16.engine \
  --enable-fp16

python3 compare_engines.py \
  --reference-report \
    outputs/precision_recovery/03_entropy_vs_minmax/evaluation/precision_evaluation.json \
  --manifest outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json \
  --int8-engine \
    outputs/precision_recovery/04_layer_sensitivity/yolov8n_minmax_detection_head_fp16.engine \
  --output-dir \
    outputs/precision_recovery/04_layer_sensitivity/detection_head_fp16_evaluation
```

The shared timing cache reduced this engine build to approximately 38 seconds, compared with roughly
15 minutes for the first uncached mixed-precision build. Engine Inspector reported 21 fused compute
layers for the constrained head and no INT8 inputs, outputs, or weights within those compute layers.
The neck and backbone were not part of the profile.

Recorded accuracy result:

| Candidate | mAP50-95 | mAP50 | Precision | Recall | Wrapper mean (ms) | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MinMax, no explicit constraints | 0.3444 | 0.4892 | 0.0429 | 0.7936 | 4.060 | FAIL |
| MinMax, complete detection head FP16 | 0.3463 | 0.4897 | 0.0414 | 0.7946 | 4.208 | FAIL |

The complete FP16 detection head recovered `0.00198` mAP50-95 and `0.00041` mAP50 relative to the
unconstrained MinMax candidate. Its mAP50-95, precision, and recall passed their limits. The only
failed metric remained mAP50: its drop from PyTorch was `0.02057`, exceeding the predeclared `0.02`
limit by approximately `0.00057`.

Matched `trtexec` evidence using 120 measured iterations:

| Engine | Mean latency (ms) | P90 (ms) | GPU compute mean (ms) | Throughput (qps) |
| --- | ---: | ---: | ---: | ---: |
| FP16 reference | 2.713 | 2.728 | 1.519 | 652.1 |
| MinMax INT8 | 2.349 | 2.359 | 1.145 | 822.3 |
| MinMax, complete detection head FP16 | 2.445 | 2.457 | 1.258 | 786.8 |

The mixed engine retained approximately `9.9%` lower mean latency and `20.7%` higher throughput than
FP16 in these matched measurements, but it is not releasable because the unchanged accuracy gate
failed. Performance benefit cannot override the predeclared quality decision.

This is the final legacy-calibrator PTQ candidate. Expanding fallback into the neck would move the
engine closer to FP16 while preserving calibration and mixed-precision complexity. Further recovery
should use an explicit Q/DQ model or QAT rather than additional implicit-quantization layer profiles.

Evidence:

- `outputs/precision_recovery/04_layer_sensitivity/`
  `yolov8n_minmax_detection_head_fp16.engine.json`
- `outputs/precision_recovery/04_layer_sensitivity/`
  `yolov8n_minmax_detection_head_fp16.engine.layers.json`
- `outputs/precision_recovery/04_layer_sensitivity/detection_head_fp16_evaluation/`
  `precision_evaluation.json`
- `outputs/precision_recovery/04_layer_sensitivity/detection_head_fp16_trtexec.log`

## Step 05: ModelOpt Explicit Q/DQ PTQ

Step 05 calibrates and exports Q/DQ ONNX in `learn-tensorrt-modelopt`, then builds and evaluates the
explicitly quantized graph in the original `trt_dev` TensorRT 8.6.1 container. It does not alter or
replace the legacy-calibrator evidence from Steps 01-04. The implementation is in
`precision_recovery/05_modelopt_ptq/modelopt_ptq.py`; generated ONNX models, metadata, engines, and
logs belong under the ignored `outputs/precision_recovery/05_modelopt_ptq/` directory.

The first configuration is predeclared as NVIDIA ModelOpt `INT8_DEFAULT_CFG`, which uses `max`
calibration for INT8 activations and per-channel INT8 weights. Calibration reuses the production
letterbox/RGB/FP32 normalization path and streams small batches to the GPU. It exports a static
`images` input of `(1, 3, 640, 640)` and raw `output0` of `(1, 84, 8400)`, then requires ONNX checker
success and nonzero `QuantizeLinear` and `DequantizeLinear` node counts.

Start the persistent container and run the focused CPU tests inside it:

```bash
docker start learn-tensorrt-modelopt
docker exec learn-tensorrt-modelopt bash -lc '
  cd /workspace/Learn-TensorRT &&
  python3 -m unittest -v \
    12_yolov8_int8_calibration/precision_recovery/05_modelopt_ptq/test_modelopt_ptq.py
'
```

Run the 32-image pipeline smoke calibration before the formal candidate:

```bash
docker exec learn-tensorrt-modelopt bash -lc '
  cd /workspace/Learn-TensorRT &&
  python3 12_yolov8_int8_calibration/precision_recovery/05_modelopt_ptq/modelopt_ptq.py \
    --candidate-kind smoke \
    --calibration-images 32 \
    --batch-size 4 \
    --name yolov8n_modelopt_int8_max_smoke32
'
```

The smoke artifact validates conversion and explicit Q/DQ export only. Its metadata sets
`valid_for_accuracy_gate` to `false`; do not evaluate or report it as a PTQ accuracy result.

After the smoke path succeeds, generate the primary formal candidate with all 3,000 images from the
coverage-aware manifest:

```bash
docker exec learn-tensorrt-modelopt bash -lc '
  cd /workspace/Learn-TensorRT &&
  python3 12_yolov8_int8_calibration/precision_recovery/05_modelopt_ptq/modelopt_ptq.py \
    --candidate-kind formal \
    --calibration-images 3000 \
    --batch-size 4 \
    --name yolov8n_modelopt_int8_max_train3000
'
```

The formal candidate is built in `trt_dev` with `--int8 --fp16`. TensorRT 8.6 requires the INT8
builder flag even though Q/DQ nodes already provide the quantization scales; its log confirms that
the legacy calibrator is not used in explicit-precision mode. Do not provide a calibration cache,
use validation images for calibration, perform QAT, or change an accuracy threshold.

Build the optimized formal explicit Q/DQ graph with INT8 Q/DQ constraints and FP16 for eligible
high-precision layers. The wrapper uses a dedicated timing cache and records the exact `trtexec`
argument vector, complete build log, detailed Engine Inspector layer JSON, and artifact hashes:

```bash
docker exec trt_dev bash -lc '
  cd /workspace/Projects/Learn-TensorRT &&
  python3 \
    12_yolov8_int8_calibration/precision_recovery/05_modelopt_ptq/build_trt86_qdq_engine.py
'
```

Evaluate only the new candidate while reusing the identity-validated TRT8.6 PyTorch/FP32/FP16
references from Step 03. The evaluator still processes all 5,000 validation images and applies the
unchanged four regression thresholds:

```bash
docker exec trt_dev bash -lc '
  cd /workspace/Projects/Learn-TensorRT/12_yolov8_int8_calibration &&
  python3 compare_engines.py \
    --manifest outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json \
    --reference-report \
      outputs/precision_recovery/03_entropy_vs_minmax/evaluation/precision_evaluation.json \
    --int8-engine \
      outputs/precision_recovery/05_modelopt_ptq/yolov8n_modelopt_int8_max_train3000_trt86_int8_fp16.engine \
    --output-dir outputs/precision_recovery/05_modelopt_ptq/evaluation
'
```

Recorded formal result on all 5,000 validation images:

| Candidate | mAP50-95 | mAP50 | Precision | Recall | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| PyTorch reference | 0.3631 | 0.5102 | 0.0427 | 0.8097 | PASS |
| ModelOpt max Q/DQ, TRT8.6 INT8+FP16 | 0.3453 | 0.4931 | 0.0432 | 0.7998 | PASS |

The Q/DQ candidate changed mAP50-95 by `-0.01781`, mAP50 by `-0.01713`, precision by
`+0.00057`, and recall by `-0.00991` versus PyTorch. Every delta remained within the unchanged
limits of `0.02`, `0.02`, `0.03`, and `0.03`, respectively. This is the first Lesson 12 INT8
candidate to pass the complete quality gate.

Engine Inspector evidence contains both INT8 and FP16 tensor-format descriptions. FP32 remains at
the external I/O boundary and in selected high-precision tensors, constants, biases, and tactics;
the recorded keyword counts are evidence mentions, not compute-layer counts. The candidate-only
wrapper mean latency was `5.124 ms`, while the reused FP16 reference recorded `4.500 ms`. Those
wrapper measurements are diagnostic and do not replace a matched `trtexec` performance comparison.

Evidence:

- `outputs/precision_recovery/05_modelopt_ptq/yolov8n_modelopt_int8_max_train3000.onnx.json`
- `outputs/precision_recovery/05_modelopt_ptq/`
  `yolov8n_modelopt_int8_max_train3000_trt86_int8_fp16.engine.json`
- `outputs/precision_recovery/05_modelopt_ptq/`
  `yolov8n_modelopt_int8_max_train3000_trt86_int8_fp16.layers.json`
- `outputs/precision_recovery/05_modelopt_ptq/evaluation/precision_evaluation.json`

## Legacy PTQ Conclusion And Handoff

The pinned TensorRT 8.6.1 legacy-calibrator path is complete. Its strongest legacy accuracy
candidate was the MinMax engine with the complete detection head constrained to FP16. It retained
measurable performance benefit over FP16 but failed the unchanged mAP50 gate, so FP16 remained the
legacy sequence's release candidate.

ModelOpt itself was not installed into the pinned environment. Calibration and Q/DQ export ran in
the separate version-pinned ModelOpt image; TensorRT 8.6 then consumed the portable ONNX artifact
and produced the first passing INT8 quality candidate. This preserves the dependency boundary while
allowing a matched comparison against the established TensorRT 8.6 references.

Future explicit Q/DQ experiments should continue to reuse only portable source artifacts:

- `05_torch_to_onnx/outputs/yolov8n.onnx` as the starting FP32 graph;
- the versioned calibration manifest and its image hashes;
- the unchanged val2017 manifest, labels, postprocessing settings, and regression thresholds;
- the saved PyTorch/FP32/FP16 reference report when every recorded software and artifact identity
  still matches.

Do not reuse TensorRT engines, calibration tables, or tactic timing caches across TensorRT, CUDA,
GPU architecture, or container changes. An explicit Q/DQ or QAT export is a new ONNX model artifact
and must receive a new hash, engine, performance report, and complete gate result.

## Sequence Status

1. Preprocessing parity.
2. Versioned calibration-set coverage experiment: completed, gate failed.
3. Entropy versus MinMax calibration: completed; MinMax nearly passed but failed mAP50.
4. Layer-sensitivity and explicit mixed-precision constraints: complete detection-head FP16 was the
   final legacy-PTQ candidate; it improved accuracy but failed mAP50.
5. Drift examples were captured in full four-backend reports and used as diagnostic evidence.
6. ModelOpt explicit Q/DQ PTQ: completed; the 3,000-image max-calibrated TRT8.6 INT8+FP16 candidate
   passed all four unchanged quality thresholds. QAT remained out of scope.
