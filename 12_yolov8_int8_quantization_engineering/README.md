# 12 - YOLOv8 INT8 Quantization Engineering

This advanced case study follows the engineering evolution from TensorRT legacy calibration to
ModelOpt explicit Q/DQ quantization. The objective is not to produce an INT8 engine at any cost. It
is to establish reproducible data and quality contracts, reject unqualified candidates, and deploy
INT8 only when it improves matched performance without violating quality.


## Recorded Outcome

The complete case study is generated from machine-readable evidence in
[`reports/quantization_case_study.md`](reports/quantization_case_study.md). Legacy Entropy, MinMax,
and complete-detection-head FP16 candidates failed the unchanged quality contract. ModelOpt
explicit-Q/DQ candidates passed in TensorRT 8.6 and TensorRT 10.14.

The final TensorRT 10 native-FP16 Q/DQ candidate reached `0.3454` mAP50-95 and `0.4946` mAP50, but
matched performance reached only `522.188 qps` versus `636.729 qps` for FP16. Its mean GPU compute
time was `1.898 ms` versus `1.556 ms`. FP16 therefore remains the deployment decision: INT8 quality
was recovered, but deployment value was not demonstrated.

## Engineering Questions

By the end of the lesson, the evidence must answer:

1. Why are these calibration images representative?
2. Which model, data, preprocessing, postprocessing, metric, runtime, and engines define the
   reference?
3. Which single variable changes in each candidate experiment?
4. Does the candidate pass the predeclared task-quality gate?
5. If it passes, is it faster than the version-matched FP16 reference?
6. Is the extra quantization complexity justified for deployment?

## Fixed Data And Quality Contract

Validation uses the complete 5,000-image COCO val2017 split with human labels. Calibration and
validation must have no byte-identical image overlap. The release thresholds and evaluation
settings are declared in `configs/quality_contract.json` before candidate results are inspected.
`compare_engines.py` loads confidence, NMS, maximum detections, input shape, metric ID, and every
allowed regression directly from this file; these values are not duplicated as CLI defaults. Every
evaluation records the contract hash, and reference reuse rejects a different contract.

The canonical calibration set is a new, independently selected 3,000-image split. It is selected
from a fixed 5,000-image train2017 candidate pool using category seeding followed by deterministic
farthest-point coverage over object scale, image and box geometry, luminance, contrast, saturation,
and edge density.

Prepare the new manifest and materialize its selected images from the already downloaded local
candidate pool:

```bash
python3 12_yolov8_int8_quantization_engineering/tools/prepare_calibration_dataset.py \
  --materialize
```

The script writes ignored raw evidence under `outputs/data_preparation/`, the canonical manifest to
`data/dataset_manifest.json`, and the fixed 5,000-image candidate-pool identity plus selected IDs to
`data/calibration_selection.json`. Review and commit both data files only after the selection is
final. Re-running with identical candidate metadata, annotations, and algorithm version must
produce the same selected IDs and hashes.

## Reference Bundles

PyTorch, TensorRT FP32, and TensorRT FP16 are evaluated once per immutable environment identity.
Every later INT8 candidate runs alone and reuses that reference bundle. The bundle identity covers:

- model weights and ONNX hashes;
- validation manifest and quality-contract hashes;
- preprocessing, postprocessing, and metric implementation IDs;
- TensorRT/CUDA/GPU runtime identity;
- FP32 and FP16 engine hashes.

Changing an INT8 engine, calibration cache, calibration algorithm, or precision profile does not
invalidate the reference. Changing any reference identity does. In particular, a TensorRT 8.6
reference can never be reused for a TensorRT 10 experiment.

Generated bundles belong under:

```text
outputs/references/trt86_full/reference_bundle.json
```

After the one complete evaluation, create the bundle with:

```bash
python3 12_yolov8_int8_quantization_engineering/tools/create_reference_bundle.py \
  --report <full-reference-report.json> \
  --onnx 05_torch_to_onnx/outputs/yolov8n.onnx \
  --quality-contract \
    12_yolov8_int8_quantization_engineering/configs/quality_contract.json \
  --output \
    12_yolov8_int8_quantization_engineering/outputs/references/trt86_full/reference_bundle.json
```

The existing evaluator's candidate-only mode is the required execution model: first generate one
complete reference report and its bundle, then pass `--reference-bundle` and the declared
`--experiment-id` for each candidate. The evaluator validates the bundle identity, experiment
matrix, engine build metadata, and source-model metadata before inference. Do not rerun PyTorch,
FP32, and FP16 merely because a new INT8 candidate was built.

## Experiment Sequence

The declared matrix is `configs/experiments.json`. Each stage changes one quantization decision.

### 0. Qualification And Reference

- verify dataset hashes and calibration/validation separation;
- prove calibration and evaluation preprocessing are byte-identical;
- build version-matched FP32 and FP16 engines;
- evaluate PyTorch, FP32, and FP16 once;
- save the immutable reference bundle.

Preprocessing parity is a prerequisite, not an accuracy-recovery experiment.

```bash
python3 tools/verify_preprocessing_parity.py
```

### 1. Legacy Entropy PTQ

Build a TensorRT 8.6 entropy-calibrated INT8 engine from the canonical 3,000-image manifest. Record
the calibration algorithm, cache identity, timing cache, builder flags, engine hash, Inspector
evidence, and task-quality result.

### 2. Legacy MinMax PTQ

Change only the calibration algorithm to MinMax. Use distinct cache and engine identities so a
calibration table cannot be reused under the wrong algorithm.

### 3. Detection-Head Mixed-Precision Diagnosis

Keep the MinMax experiment fixed and constrain the complete YOLOv8 detection head to FP16 with
strict TensorRT precision constraints. The profile follows the prediction towers through reshape,
concatenation, DFL, box decoding, class sigmoid, and final output assembly. The builder validates the
expected 67-layer data-flow structure, and the Engine Inspector must prove that every internal head
output is FP16; only the explicit external FP32 output boundary is allowed.

This is the only layer-sensitivity candidate in the formal course. The historical box-only and
box-plus-class-output experiments are intentionally omitted because they add procedural noise
without changing the engineering conclusion.

### 4. ModelOpt Explicit Q/DQ

Legacy calibration leaves scale placement largely to TensorRT. This stage moves quantization intent
into the model graph with ModelOpt `QuantizeLinear`/`DequantizeLinear` nodes and builds an
INT8+FP16 engine. It uses a second image and environment because the pinned TensorRT 8.6 development
container does not provide the required ModelOpt/TensorRT 10 toolchain.

The local environments used for this course are recorded in `configs/environments.json`:

- `trt_dev`: `nvcr.io/nvidia/tensorrt:23.10-py3`, TensorRT 8.6.1 and CUDA 12.2;
- `learn-tensorrt-modelopt`: `nvcr.io/nvidia/pytorch:25.11-py3`, TensorRT 10.14.1.48,
  CUDA 13.0, ModelOpt 0.37.0, and Ultralytics 8.4.22.

The ModelOpt environment can be reconstructed approximately as follows. Pin exact package versions
in the experiment metadata rather than silently accepting future latest releases:

```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3

docker run -d --gpus all \
  --name learn-tensorrt-modelopt \
  --ipc=host \
  -v "$PWD:/workspace/Learn-TensorRT" \
  nvcr.io/nvidia/pytorch:25.11-py3 sleep infinity

docker exec learn-tensorrt-modelopt bash -lc '
  python3 -m pip install \
    nvidia-modelopt==0.37.0 \
    ultralytics==8.4.22 \
    onnx==1.18.0
'

docker exec learn-tensorrt-modelopt bash -lc '
  python3 -c "import modelopt, tensorrt, torch, onnx, cv2, ultralytics"
'
```

The NGC image already supplies its CUDA, PyTorch, and TensorRT stack; do not replace those packages
with unrelated host or PyPI CUDA builds. Capture the actual versions after construction because an
NGC image and pip resolver can change transitive packages.

The TensorRT 8.6 FP32-high-precision Q/DQ candidate and the TensorRT 10 native-FP16 Q/DQ candidate
remain consecutive stages of the same course. Switching runtime, however, creates a hard evidence
boundary: build new TensorRT 10 FP32 and FP16 engines and generate a new TensorRT 10 reference
bundle before evaluating the native-FP16 Q/DQ candidate.

### 5. Deployment Decision

Only quality-passing candidates are eligible for matched performance testing. Compare the selected
Q/DQ INT8+FP16 candidate against the FP16 reference built in the same container and TensorRT
version. Use identical shapes, warmup, measured iterations, transfer settings, and synchronization.

The recorded case-study result is intentionally non-obvious: INT8+FP16 passed the quality gate but
was slower than pure FP16. Possible causes include:

- insufficient INT8 kernel coverage, leaving substantial FP16 or FP32 computation;
- Q/DQ boundaries introducing reformat, cast, quantize, and dequantize work;
- tensor layouts or channel dimensions that prevent efficient INT8 tactics;
- mixed-precision boundaries increasing memory traffic and synchronization;
- tactic selection choosing slower kernels for this GPU and batch size;
- the model being too small or batch-1 execution being too latency-bound to amortize conversions.

A follow-up investigation could compare Engine Inspector layer precision and formats, count Q/DQ-
origin reformats, inspect `trtexec --dumpProfile` layer times, compare tactic choices, use Nsight
Systems for launch/transfer gaps, use Nsight Compute for kernel utilization, and test supported
batch sizes. That root-cause investigation is deliberately outside this lesson. The engineering
decision remains: passing accuracy makes a candidate eligible; it does not make it deployable.

## Reproduction

The complete commands and output layout are documented in
[`docs/reproduction.md`](docs/reproduction.md). The important execution rule is that TRT8
candidates reuse the TRT8 reference, while the runtime change requires one new full TRT10
reference evaluation before any TRT10 candidate reuse.

## Deliverables

- versioned 3,000-image calibration manifest and coverage report;
- one reference bundle for TensorRT 8.6 and one for TensorRT 10;
- Entropy, MinMax, detection-head FP16, and Q/DQ candidate metadata;
- machine-readable quality reports with unchanged gates;
- matched FP16 versus passing Q/DQ performance evidence;
- `reports/quantization_case_study.md` containing the final decision.

Generated datasets, engines, calibration tables, timing caches, predictions, and raw benchmark
captures stay in ignored output directories.

## CPU Verification

Run inside `trt_dev` so NumPy and OpenCV match the course environment:

```bash
PYTHONPATH=12_yolov8_int8_quantization_engineering \
python3 -m unittest discover \
  -s 12_yolov8_int8_quantization_engineering/tests -v
```

Run ModelOpt and TensorRT 10 focused tests inside `learn-tensorrt-modelopt`:

```bash
PYTHONPATH=12_yolov8_int8_quantization_engineering \
python3 -m unittest discover \
  -s 12_yolov8_int8_quantization_engineering/modelopt \
  -p 'test_*.py' -v
```

GPU-, TensorRT-, and ModelOpt-dependent commands must run in the matching container described above.
