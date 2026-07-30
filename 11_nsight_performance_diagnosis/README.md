# 11 - Nsight Performance Diagnosis

## Purpose

- Replace guesswork with timeline-based performance diagnosis.
- Profiling immediately after the first C++ pipeline gives you a baseline before optimization.
- High-end deployment roles expect evidence: latency tables, profiler traces, and bottleneck explanations.

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
of truth for CPU gaps, CUDA API blocking, memory copies, and kernel placement. The capture does not
enable Nsight's optional `--cuda-memory-usage` mode: with the pinned CUDA 13.0/Nsight 2025.5 stack
that mode crashes during TensorRT execution-context teardown, while ordinary CUDA tracing records
the allocation APIs needed by this lesson without changing application behavior.

## Prerequisites

Complete lessons 06 and 10 first:

```bash
python3 06_trtexec_engine/build_and_benchmark.py --builds static_fp32
cmake -S 10_yolov8_trt_cpp -B 10_yolov8_trt_cpp/build
cmake --build 10_yolov8_trt_cpp/build
nsys --version
```

## Deliverables

- `profile_yolov8_cpp.py` strict Nsight Systems capture workflow
- Ignored capture, SQLite, statistics, environment, and summary artifacts
- CPU-only tests for command construction and evidence gates

## Runnable Artifact

- `profile_yolov8_cpp.py`: builds lesson 10, runs an in-process steady-state benchmark, loads the
  matching lesson 06 `trtexec` timing JSON when available, captures Nsight Systems with the
  image-specific HPC-X preload removed, and writes JSON
  and Markdown reports.
- `tests/test_profile_yolov8_cpp.py`: focused tests for percentile, schema-validation, composition,
  and diagnosis behavior.

Generated files go to `outputs/`, which is ignored by git.
`diagnosis_summary.json` records the TensorRT/CUDA/GPU/driver/container identity plus the profiled
engine and image SHA-256 values so later checkpoints
can reject diagnosis evidence collected from a different engine.

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
- `outputs/diagnosis_summary.json`: machine-readable runtime identity, application, `trtexec`, and
  Nsight metadata.
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


## Appendix: Commands Executed By The Default Run

The following commands are the important subprocesses used by the default run. Run them from
`11_nsight_performance_diagnosis/` so the relative paths resolve as shown. The script locates
`nsys` through `PATH` and removes the development image's `LD_PRELOAD` only from Nsight
subprocesses because that HPC-X preload conflicts with Nsight library injection; in the pinned
development container it resolved to
`/usr/local/cuda/bin/nsys`.

Start the complete workflow:

```bash
python3 profile_yolov8_cpp.py
```

Configure and build the lesson 10 application:

```bash
cmake -S . -B build
cmake --build build -j2
```

These two commands are executed with `../10_yolov8_trt_cpp` as their working directory.

Collect the steady-state baseline with 5 warmup iterations and 50 measured iterations:

```bash
../10_yolov8_trt_cpp/build/yolov8_trt_cpp \
  --engine ../06_trtexec_engine/outputs/yolov8n_static_fp32.engine \
  --image ../assets/img.jpeg \
  --output-dir outputs/baseline_run \
  --confidence 0.25 \
  --iou 0.45 \
  --max-detections 100 \
  --warmup-iterations 5 \
  --iterations 50
```

Capture a shorter Nsight Systems trace with 2 warmup iterations and 5 measured iterations:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  --output outputs/nsys/yolov8_trt_cpp \
  ../10_yolov8_trt_cpp/build/yolov8_trt_cpp \
  --engine ../06_trtexec_engine/outputs/yolov8n_static_fp32.engine \
  --image ../assets/img.jpeg \
  --output-dir outputs/nsys/target_run \
  --confidence 0.25 \
  --iou 0.45 \
  --max-detections 100 \
  --warmup-iterations 2 \
  --iterations 5
```

Export the trace to SQLite and generate the text summaries used for diagnosis:

```bash
nsys export \
  --type sqlite \
  --force-overwrite=true \
  --output outputs/nsys/yolov8_trt_cpp.sqlite \
  outputs/nsys/yolov8_trt_cpp.nsys-rep

nsys stats \
  --force-export=true \
  --report nvtx_pushpop_sum \
  --report cuda_api_sum \
  --report cuda_gpu_sum \
  outputs/nsys/yolov8_trt_cpp.nsys-rep
```

The Python driver captures the baseline and profiler metadata in
`outputs/diagnosis_summary.json`, and redirects the `nsys stats` output to
`outputs/nsys/nsys_stats.txt` itself.
