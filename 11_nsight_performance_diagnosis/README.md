# 11 - Nsight Performance Diagnosis

This lesson profiles the lesson 10 C++ YOLOv8 TensorRT artifact without confusing first-inference
startup behavior with steady-state latency.

Goal: use repeated in-process measurements, a model-only `trtexec` reference, NVTX ranges, and an
Nsight Systems timeline to explain where request latency is spent.

Topics:

- warmup versus measured iterations
- P50/P90/P99 latency reporting
- host `enqueueV3` time versus GPU compute time
- per-request CPU, transfer, GPU, and unaccounted latency shares
- `trtexec` model-only comparison
- Nsight Systems command-line capture and text statistics
- NVTX-guided CPU/GPU timeline reading
- synchronization gaps and GPU starvation

## Why This Matters

A new TensorRT execution context can have an expensive first inference. Launching a new process for
every sample repeatedly measures that cold path and can produce a false bottleneck diagnosis.

This lesson instead runs the pipeline repeatedly inside one C++ process:

```text
engine/context/stream initialization
  -> warmup iterations (reported separately)
  -> measured iterations (steady-state percentiles)
  -> visualization and report writing
```

The automatic diagnosis is deliberately a heuristic. The Nsight timeline remains the final source
of truth for CPU gaps, CUDA API blocking, memory copies, and kernel placement.

## Runnable Artifact

- `profile_yolov8_cpp.py`: builds lesson 10, runs an in-process steady-state benchmark, loads the
  matching lesson 06 `trtexec` timing JSON when available, captures Nsight Systems, and writes JSON
  and Markdown reports.
- `tests/test_profile_yolov8_cpp.py`: focused tests for percentile, schema-validation, composition,
  and diagnosis behavior.

Generated files go to `outputs/`, which is ignored by git.

## Prerequisites

Complete lessons 06 and 10 in the pinned TensorRT development container:

```bash
python3 06_trtexec_engine/build_and_benchmark.py --builds static_fp32
cmake -S 10_yolov8_trt_cpp -B 10_yolov8_trt_cpp/build
cmake --build 10_yolov8_trt_cpp/build
nsys --version
```

## Run

Run from this lesson directory:

```bash
python3 profile_yolov8_cpp.py
```

Defaults:

- 5 warmup iterations and 50 measured baseline iterations
- 2 warmup iterations and 5 measured iterations in the Nsight capture
- P99 warning when fewer than 100 measured baseline samples are used

Run a fast smoke diagnosis without Nsight:

```bash
python3 profile_yolov8_cpp.py \
  --warmup-iterations 2 \
  --iterations 5 \
  --skip-nsys
```

Collect a more stable tail-latency sample:

```bash
python3 profile_yolov8_cpp.py --warmup-iterations 10 --iterations 200
```

Profile another engine or shorten the trace:

```bash
python3 profile_yolov8_cpp.py \
  --engine ../06_trtexec_engine/outputs/yolov8n_static_fp16.engine \
  --nsys-warmup-iterations 1 \
  --nsys-iterations 3
```

Skip rebuilding lesson 10 when it is already current:

```bash
python3 profile_yolov8_cpp.py --skip-build
```

## Outputs

- `outputs/baseline_run/detections.json`: lesson 10 warmup and measured samples.
- `outputs/diagnosis_summary.json`: machine-readable application, `trtexec`, and Nsight metadata.
- `outputs/diagnosis_report.md`: steady-state tables and diagnosis notes.
- `outputs/nsys/yolov8_trt_cpp.nsys-rep`: Nsight Systems timeline.
- `outputs/nsys/yolov8_trt_cpp.sqlite`: exported trace database.
- `outputs/nsys/nsys_stats.txt`: NVTX, CUDA API, and GPU summary reports.

The latency table includes:

- `preprocess` and `postprocess`: CPU wall-clock stages.
- `enqueue_host`: CPU time spent calling `enqueueV3`.
- `gpu_compute`: CUDA-event time spanning TensorRT GPU work.
- `h2d` and `d2h`: CUDA-event copy times.
- `total`: preprocessing through postprocessing.

`enqueue_host` and `gpu_compute` overlap conceptually and are not added together in the request
composition. Engine deserialization, image decoding, visualization, file writing, and process
startup are outside the steady-state total.

## Reading The Timeline

Open the report:

```bash
nsys-ui outputs/nsys/yolov8_trt_cpp.nsys-rep
```

Use the NVTX ranges to navigate:

- `warmup_iteration_0`: compare first-inference GPU work with later iterations.
- `measured_iteration_*`: inspect representative steady-state requests.
- `preprocess` and `postprocess`: identify CPU-heavy regions.
- `h2d_submit`, `tensorrt_enqueue_host`, and `d2h_submit_and_wait`: relate CUDA API submission and
  synchronization to the CUDA HW row.

Then verify whether H2D, TensorRT kernels, and D2H execute in order, whether the CPU creates gaps,
and whether synchronization leaves the GPU idle between requests.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests cover nearest-rank percentiles, invalid report values, per-request composition, and the
minimum lead required before the report declares one category dominant.

## Checkpoints

- Compare the first warmup `gpu_compute` value with steady-state P50.
- Compare application `gpu_compute` with the matching lesson 06 `trtexec` reference.
- Run FP32 and FP16 engines and identify which stages actually change.
- Explain why 10 samples cannot support a stable P99 claim.
- Find one measured iteration in Nsight and account for its CPU work, copies, kernels, and gaps.
- Try one optimization and cite before-and-after reports and timelines.

Acceptance criteria:

- Cold first-inference behavior is separated from steady-state samples.
- P50/P90/P99 are generated from repeated in-process measurements.
- Host enqueue time is not mislabeled as GPU compute time.
- Nsight captures contain named NVTX iteration and pipeline ranges.
- The report compares the application with a model-only reference when timing JSON is available.
- Any bottleneck claim is presented as a heuristic and checked against timeline evidence.
