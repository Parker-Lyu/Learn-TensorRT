# 31 - Evidence-Driven CUDA Kernel Optimization

## Purpose

Complete an evidence-driven CUDA optimization cycle on Lesson 20's BGR-to-RGB NCHW conversion:
establish a two-kernel teaching baseline, diagnose its launch and memory costs, test a fusion
hypothesis, validate the result, investigate additional candidates, and decide whether further work
is justified in the production preprocessing path.

The lesson separates three conclusions that are often confused:

- whether a controlled kernel change is faster;
- whether the already-fused GPU preprocessing path has another useful optimization;
- whether any measured change affects end-to-end deployment performance.

## Prerequisites

- Complete Lesson 13 so system-level profiling and measurement boundaries are familiar.
- Complete and build Lesson 20; its OpenCV comparison and error limits remain the production
  correctness contract.
- Use the persistent pinned development container with an accessible NVIDIA GPU. Nsight Compute
  hardware counters can be disabled by the host driver's profiling-permission policy.
- Build Lesson 21 and retain its metrics only when adding production-pipeline context to the report.
  A single pipeline run is not an optimization A/B comparison.

## Deliverables

- A C++17/CUDA standalone benchmark with fixed input, explicit warmup, repeated CUDA event timing,
  NVTX ranges, and exact correctness checks
- A two-launch `unfused` teaching baseline and a one-launch `baseline_16x16` fused implementation
- Three follow-up candidates covering block shape, linear indexing, and four-pixel vectorized reads
- Reproducible Nsight Systems and Nsight Compute capture commands
- Generated benchmark JSON, `.nsys-rep`, `.ncu-rep`, metric summary, environment manifest, and a
  decision report with separate kernel, preprocessing, and deployment boundaries

## Design

The `unfused` variant first converts packed BGR bytes into an RGB HWC float intermediate, then
launches a second kernel to reorder HWC into CHW. The fused variant writes normalized CHW output
directly. This controlled pair makes the hypothesis explicit: removing one launch and the
intermediate global-memory round trip should reduce standalone execution time, but measurement still
decides whether the hypothesis holds on the current GPU.

For the default 640x640 input, the intermediate contains 1,228,800 floats, or 4.6875 MiB. The second
kernel must read that intermediate after the first kernel writes it. Fusion removes both transfers
and the second launch while preserving the final CHW output writes.

The remaining variants begin from the fused algorithm. They test whether a different block shape,
linear indexing, or vectorized input reads improves it. A plausible low-level technique is not
accepted unless matched timing improves and the exact numerical contract still passes.

The unprofiled standalone process owns performance timing. Nsight Compute may replay kernels several
times to collect counters, so profiler-reported duration is never benchmark evidence. Allocations,
input generation, CUDA context initialization, and host-device setup remain outside measured CUDA
event intervals.

## Build

Run from the repository root inside the persistent development container:

```bash
cmake -S 20_cuda_preprocess_npp -B 20_cuda_preprocess_npp/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build 20_cuda_preprocess_npp/build --parallel
cmake -S 31_nsight_compute_kernel_analysis -B 31_nsight_compute_kernel_analysis/build \
  -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build 31_nsight_compute_kernel_analysis/build --parallel
```

## Run

Measure the two-launch teaching baseline before inspecting other results:

```bash
./31_nsight_compute_kernel_analysis/build/lesson31_kernel_benchmark \
  --variant unfused --warmup 50 --iterations 500 \
  --output 31_nsight_compute_kernel_analysis/outputs/unfused_benchmark.json
```

Use its two launches and intermediate HWC float buffer to form the fusion hypothesis. Then measure
the fused implementation under the same conditions:

```bash
./31_nsight_compute_kernel_analysis/build/lesson31_kernel_benchmark \
  --variant baseline_16x16 --warmup 50 --iterations 500 \
  --output 31_nsight_compute_kernel_analysis/outputs/fused_benchmark.json
```

Do not compare the two files if their input dimensions, warmup, iteration count, or environment
differ. Run the complete workflow to create one matched benchmark, collect Lesson 20 timing and its
Nsight Systems timeline, and capture one Nsight Compute report per controlled variant:

```bash
python3 31_nsight_compute_kernel_analysis/profile_kernels.py
```

When `/proc/driver/nvidia/params` reports `RmProfilingAdminOnly: 1`, the persistent container must
have the `SYS_ADMIN` capability documented in `00_environment_check/agent_env_setup.md`. Start the
workflow from the host as container root, then restore output ownership:

```bash
docker exec --user root learn-tensorrt bash -lc \
  'cd /workspace/Learn-TensorRT && python3 31_nsight_compute_kernel_analysis/profile_kernels.py && \
   chown -R 1000:1000 31_nsight_compute_kernel_analysis/outputs 20_cuda_preprocess_npp/outputs reports'
```

The pinned image is built with the host user's UID/GID, which is `1000:1000` in the documented
course container. Use the actual configured UID/GID if the image was built differently.

The workflow removes the container's `LD_PRELOAD` only for profiler subprocesses, matching Lesson
13's Nsight injection requirement. It collects `LaunchStats`, `Occupancy`, `SpeedOfLight`,
`MemoryWorkloadAnalysis`, `SchedulerStats`, and `WarpStateStats`.

Read the evidence in this order:

1. Confirm that every variant satisfies the exact CPU-reference comparison.
2. Compare unprofiled CUDA event timing for `unfused` and `baseline_16x16`.
3. Use their Nsight Compute reports to explain launch count, bytes moved, occupancy, scheduler
   activity, and dominant warp stalls.
4. Compare `block_32x8`, `linear`, and `vectorized` against the fused 16x16 kernel. Reject candidates
   that only improve an isolated counter.
5. Use Lesson 20 timing to determine the fused conversion kernel's share of the production
   preprocessing path. Mapped host memory is a separate transfer strategy, not a device-memory
   kernel baseline.
6. Require a matched production implementation and A/B measurement before claiming complete
   preprocessing or end-to-end improvement.

To add context from a completed Lesson 21 run:

```bash
python3 31_nsight_compute_kernel_analysis/profile_kernels.py \
  --pipeline-metrics 21_integrated_tensorrt_video_pipeline/output/metrics.json
```

This records the scale of pipeline preprocessing and inference work. It does not turn the standalone
variant comparison into a pipeline A/B experiment.

Generate the local decision report:

```bash
python3 31_nsight_compute_kernel_analysis/generate_report.py
```

## Outputs

- `outputs/unfused_benchmark.json` and `outputs/fused_benchmark.json` support the staged exercise.
- `outputs/kernel_benchmark.json` contains the matched five-variant CUDA event timing and exact
  errors used by the report.
- `outputs/profile_manifest.json` records commands, tool versions, GPU/driver/clocks, and evidence
  availability.
- `outputs/nsys/*.nsys-rep` and `outputs/ncu/*.ncu-rep` are environment-specific profiler captures.
- `outputs/ncu_metrics_summary.json` is regenerated from raw Nsight Compute metric rows.
- `reports/31_nsight_compute_kernel_analysis.md` separately reports the controlled fusion result,
  candidate decision, Lesson 20 production context, and end-to-end evidence boundary.

All generated evidence remains ignored. The repository commits the code and commands needed to
regenerate it on the target environment; a report from one GPU is not a portable performance claim.

## Tests

```bash
ctest --test-dir 31_nsight_compute_kernel_analysis/build --output-on-failure
```

The CUDA test runs every variant on dimensions with vectorized tail handling and compares every
element with the CPU reference. Python tests cover command construction, raw metric parsing,
correctness gates, fusion and candidate decisions, and the rule that a kernel-only result is not a
deployment result. The CUDA case is skipped when no device is accessible; actual profiler capture
additionally requires hardware-counter permission.

## Checkpoints

1. Before measuring the fused kernel, predict which launch and memory operations fusion removes.
2. Defend or reject the fusion hypothesis using matched timing and correctness evidence.
3. Explain why the vectorized-read result follows from the combined register, occupancy, memory,
   scheduler, and warp-stall evidence rather than from its name.
4. Identify which candidate ideas failed or remained inconclusive and why they should not ship.
5. Report standalone kernel, complete GPU preprocessing, and pipeline conclusions separately.
6. State what additional implementation and matched evidence would be required to claim an
   end-to-end deployment gain.
