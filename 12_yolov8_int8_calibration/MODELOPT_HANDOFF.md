# ModelOpt INT8 PTQ Handoff

## Plan Override And Completed Result (2026-07-17)

The user explicitly changed the build and evaluation target after this handoff was written:

- Keep ModelOpt calibration and Q/DQ ONNX export in `learn-tensorrt-modelopt`.
- Build the optimized explicit Q/DQ engine in the original `trt_dev` TensorRT 8.6.1 container.
- Use `--int8 --fp16`; TensorRT 8.6 requires the INT8 builder flag even for explicit Q/DQ, while
  its log confirms that no legacy calibrator is used.
- Reuse the identity-validated TensorRT 8.6 PyTorch/FP32/FP16 reference report and evaluate the new
  candidate on the unchanged 5,000-image validation split.

This revised plan is complete. The 3,000-image ModelOpt max candidate passed all four gates:

- mAP50-95: `0.3452849819`, delta versus PyTorch `-0.0178097979`
- mAP50: `0.4930931139`, delta `-0.0171335752`
- precision: `0.0432487458`, delta `+0.0005652262`
- recall: `0.7997798266`, delta `-0.0099078024`

Matched performance with 500 ms warmup and 120 measured iterations showed `618.236 qps` for the
Q/DQ candidate versus `650.348 qps` for FP16. The candidate is valid on quality but slower than
FP16, so FP16 remains the deployment choice while a reduced-FP32 Q/DQ export is investigated.

Primary evidence is under `outputs/precision_recovery/05_modelopt_ptq/`. The corrected reproducible
commands and decision record are in `precision_recovery/README.md`. The older TensorRT 10 plan below
is retained only as historical handoff context and must not override this user-approved change.

This handoff transfers Lesson 12 ModelOpt work to a new Codex window. Read the repository-root
`AGENTS.md` and this file before making changes.

## Objective

Produce a reproducible YOLOv8n INT8 model with NVIDIA ModelOpt PTQ, export an explicit Q/DQ ONNX
model, build it with TensorRT 10.14, and pass the existing Lesson 12 accuracy gate on the fixed
5,000-image COCO val2017 split.

This work is PTQ-only. There is no complete labeled training set, so do not perform QAT, training,
fine-tuning, or pseudo-label training. Do not relax an accuracy threshold after seeing a result.

## Container

Use the existing persistent container:

```bash
docker start learn-tensorrt-modelopt
docker exec -it learn-tensorrt-modelopt bash
cd /workspace/Learn-TensorRT
```

The repository is bind-mounted at `/workspace/Learn-TensorRT`. Run all ModelOpt, CUDA, TensorRT,
ONNX, and GPU-dependent work inside this container.

Recorded environment:

- Image: `nvcr.io/nvidia/pytorch:25.11-py3`
- GPU: NVIDIA GeForce RTX 2060, 6 GiB
- Driver: 580.159.04
- CUDA runtime: 13.0
- PyTorch: `2.10.0a0+b558c986e8.nv25.11`
- ModelOpt: `0.37.0`
- TensorRT: `10.14.1.48`
- ONNX: `1.18.0`
- ONNX Runtime: `1.23.2`
- Ultralytics: `8.4.22`
- OpenCV: `4.13.0`
- NumPy: `2.1.0`
- Codex CLI: `0.144.5`

The container has `/root/.codex/auth.json` and `/root/.codex/config.toml`; both were copied from
the host and intentionally have mode `664` per the user's request.

## Verified Baseline

Environment smoke tests passed before this handoff:

- ModelOpt imports and exposes `quantize` and `INT8_DEFAULT_CFG`.
- TensorRT, ONNX, ONNX Runtime, CUDA Python bindings, Ultralytics, and OpenCV import successfully.
- TensorRT `trtexec` sees TensorRT 10.14.1.
- The GPU is visible inside the container.
- `assets/yolov8n.pt` loads and runs on CUDA.
- Fixed-input PyTorch output shape is `(1, 84, 8400)`.
- Compared with `05_torch_to_onnx/outputs/pytorch_raw_output.npy`, the new-container output had
  maximum absolute error `0.00115966796875`, mean absolute error approximately `1.08e-6`, and
  passed `numpy.allclose` with `rtol=1e-4`, `atol=1e-4`.
- The Git worktree was clean before this handoff file was added. No ModelOpt implementation or
  quantized model has been created yet.

PyTorch emits a warning that the RTX 2060 is compute capability 7.5 and that this NGC PyTorch build
lists CUDA 12.6/12.8 configurations. CUDA inference nevertheless completed successfully. Preserve
the warning in reproducibility notes; do not claim it is harmless beyond the completed smoke test.

## Existing Data

There are two relevant calibration manifests:

1. Canonical 1,000-image manifest:
   `assets/coco/data/dataset_manifest.json`
2. Coverage-aware 3,000-image manifest:
   `12_yolov8_int8_calibration/outputs/precision_recovery/02_calibration_coverage/dataset_manifest.json`

The 3,000-image set retains the original 1,000 images and adds 2,000 images selected for broader
category, object-size, aspect-ratio, luminance, contrast, saturation, and edge-density coverage.
Its preprocessing parity check has already passed.

ModelOpt does not require forwarding an entire dataset for calibration; representative coverage is
more important than a fixed image count. For this lesson:

- Use only 32-64 calibration images for an early pipeline smoke test. Do not report its accuracy as
  a valid PTQ result.
- Use the coverage-aware 3,000-image manifest for the first and primary formal ModelOpt candidate.
- Do not build a formal 1,000-image candidate by default.
- Build the 1,000-image comparison only if evidence about calibration-data quantity is later needed.
- Load calibration images in small streaming batches appropriate for the 6 GiB GPU; do not retain
  the complete dataset in GPU memory.

Calibration images do not need labels. Never use the 5,000 validation images as calibration data.

## Fixed Accuracy Contract

Use the same complete 5,000-image COCO val2017 split, decoding, confidence threshold, NMS threshold,
and maximum-detection setting already implemented by Lesson 12.

The existing regression thresholds are:

- Maximum mAP50-95 drop: `0.02`
- Maximum mAP50 drop: `0.02`
- Maximum precision drop: `0.03`
- Maximum recall drop: `0.03`

The course metric is the documented COCO-like 101-point metric, not official `pycocotools` COCO AP.
Do not change the metric implementation or claim it is the official COCO score.

Existing baseline report:
`12_yolov8_int8_calibration/outputs/precision_evaluation.json`.

Existing best legacy-calibrator result:

- MinMax plus complete detection head in FP16
- mAP50-95: approximately `0.3463`
- mAP50: approximately `0.4897`
- Failed only because its mAP50 drop was approximately `0.02057`, versus the allowed `0.02`

FP16 remains the current release decision until a new candidate passes every gate.

## Corrected Execution Plan

1. Confirm the container, imports, GPU, model hashes, manifests, ignored output paths, and worktree.
2. Add a focused ModelOpt PTQ module under Lesson 12 or a new ordered recovery step. Keep the public
   API narrow and separate data loading, quantization, export, build, and metadata responsibilities.
3. Reuse the production letterbox/RGB/normalization/NCHW preprocessing path. Add focused CPU tests
   for manifest selection, input validation, batching, and metadata where practical.
4. Run a 32-64-image smoke calibration to validate ModelOpt conversion and explicit Q/DQ export.
5. Inspect the smoke ONNX graph for `QuantizeLinear` and `DequantizeLinear` nodes and validate its
   input/output identity. Do not run the complete validation gate on the smoke candidate.
6. Run formal PTQ using the coverage-aware 3,000-image manifest. Prefer ModelOpt's supported INT8
   configuration and a predeclared calibration algorithm; record all overrides and package versions.
7. Export a static `(1, 3, 640, 640)` explicit Q/DQ ONNX model and write artifact hashes plus
   quantization metadata.
8. Build the engine with TensorRT 10.14. Record the full command, log, engine hash, layer/precision
   Inspector evidence, and any FP16 fallbacks.
9. Adapt evaluation for TensorRT 10 without invalidating the old TensorRT 8.6 evidence. TensorRT
   engines are version-specific, so build new TensorRT 10 FP32/FP16 references if the evaluator
   requires matched runtime identities. Do not attempt to deserialize the old 8.6 engines in 10.14.
10. Evaluate the formal candidate on the unchanged 5,000-image validation split and apply all four
    original thresholds.
11. If it fails, perform only predeclared PTQ recovery: supported calibration choices, quantizer
    configuration, and mixed FP16/INT8 based on unlabeled calibration-output sensitivity. Do not do
    QAT and do not loosen the gate.
12. When a candidate passes, measure matched FP16/INT8 performance, generate a reproducible report,
    run focused tests and `git diff --check`, and document exactly what was and was not verified.

## Candidate Selection Discipline

Avoid repeatedly tuning against validation labels. Use unlabeled calibration images and FP32 raw
outputs for quantizer or layer sensitivity where possible. Predeclare a small ordered set of formal
candidates before running the complete validation gate. Preserve every evaluated candidate's
configuration and result, including failures.

Select the passing candidate with the fewest FP16 fallbacks. Performance cannot override a failed
quality gate.

## Output And Repository Rules

- Put generated ONNX models, TensorRT engines, raw calibration captures, logs, and local benchmark
  outputs under ignored Lesson 12 output directories.
- Commit only intentional scripts, tests, concise documentation, small manifests, and reproducibility
  metadata.
- Do not commit serialized TensorRT engines unless explicitly requested.
- Do not create practice folders, TODO-only lesson copies, branches, or solution branches.
- Preserve existing user changes and inspect `git status` before editing.
- Use the pinned legacy TensorRT container only for reproducing old evidence; use
  `learn-tensorrt-modelopt` for the new ModelOpt/TensorRT 10 experiment.

## Suggested First Commands In The New Window

```bash
cd /home/parker/Projects/Learn-TensorRT
cat AGENTS.md
cat 12_yolov8_int8_calibration/MODELOPT_HANDOFF.md
git -c safe.directory=/home/parker/Projects/Learn-TensorRT status --short
docker start learn-tensorrt-modelopt
docker exec learn-tensorrt-modelopt bash -lc '
  cd /workspace/Learn-TensorRT &&
  python3 -c "import modelopt.torch.quantization, tensorrt, torch, cv2, ultralytics" &&
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
'
```

After these checks, begin with the preprocessing/data-loader implementation and smoke calibration.
