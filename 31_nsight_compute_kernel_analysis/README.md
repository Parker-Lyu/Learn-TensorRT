# 31 - Nsight Compute Kernel Analysis

## Purpose

Use Nsight Systems and Nsight Compute evidence to decide whether optimizing Lesson 20's
`bgr_to_rgb_nchw` CUDA kernel is worthwhile. The lesson does not promise a speedup. Retaining the
baseline, or stopping because the custom kernel is not a material system bottleneck, is a valid
engineering result.

## Prerequisites

- Complete Lesson 13 so timeline selection precedes kernel-level investigation.
- Complete and build Lesson 20; its OpenCV comparison and error limits remain the correctness
  contract.
- Use the persistent pinned development container with an accessible NVIDIA GPU. Nsight Compute
  hardware counters can also be disabled by the host driver's profiling-permission policy.
- Build Lesson 21 and its dynamic TensorRT engine only when collecting matched end-to-end evidence.

## Deliverables

- A C++17/CUDA standalone benchmark with fixed input, explicit warmup, repeated CUDA event timing,
  NVTX ranges, and exact correctness checks
- Five controlled variants: baseline `16x16`, `32x8`, linear indexing, four-pixel vectorized reads,
  and an unfused two-kernel implementation
- Reproducible Nsight Systems and Nsight Compute capture commands
- Generated benchmark JSON, `.nsys-rep`, `.ncu-rep`, metric summary, environment manifest, and
  decision report

## Design

The unprofiled standalone process owns performance timing. Nsight Compute may replay kernels several
times to collect counters, so profiler-reported duration is never used as benchmark evidence.
Allocations, input generation, the first CUDA context initialization, and host-device setup are
outside measured CUDA event intervals.

The vectorized variant reads four packed BGR pixels as three aligned `uchar4` values when row layout
permits it and uses a scalar tail otherwise. The unfused variant writes an RGB HWC float intermediate
and then reorders it to CHW, making the extra launch and global-memory traffic explicit. These are
small explanatory changes, not an unrestricted parameter search.

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

Establish the unprofiled baseline first:

```bash
./31_nsight_compute_kernel_analysis/build/lesson31_kernel_benchmark \
  --variant all --warmup 50 --iterations 500 \
  --output 31_nsight_compute_kernel_analysis/outputs/kernel_benchmark.json
```

Collect the complete Lesson 20 timing, Nsight Systems timeline, and one Nsight Compute report per
variant:

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

The script removes the container's `LD_PRELOAD` only for profiler subprocesses, matching Lesson 13's
known Nsight injection requirement. It collects `LaunchStats`, `Occupancy`, `SpeedOfLight`,
`MemoryWorkloadAnalysis`, `SchedulerStats`, and `WarpStateStats`. Use the generated Systems timeline
to decide whether the conversion kernel is a material candidate. If it is not, record that decision;
the Compute exercise remains a controlled microarchitectural case, not a claimed pipeline hotspot.

To include matched end-to-end evidence from a completed Lesson 21 run:

```bash
python3 31_nsight_compute_kernel_analysis/profile_kernels.py \
  --pipeline-metrics 21_integrated_tensorrt_video_pipeline/output/metrics.json
```

Generate the local decision report:

```bash
python3 31_nsight_compute_kernel_analysis/generate_report.py
```

## Outputs

- `outputs/kernel_benchmark.json` contains unprofiled CUDA event timing and exact errors.
- `outputs/profile_manifest.json` records commands, tool versions, GPU/driver/clocks, and evidence
  availability.
- `outputs/nsys/*.nsys-rep` and `outputs/ncu/*.ncu-rep` are environment-specific profiler captures.
- `outputs/ncu_metrics_summary.json` is regenerated from raw Nsight Compute metric rows.
- `reports/31_nsight_compute_kernel_analysis.md` is the ignored generated decision report.

All generated evidence remains ignored. The repository commits the code and commands needed to
regenerate it on the target environment; a report from one GPU is not a portable performance claim.

## Tests

```bash
ctest --test-dir 31_nsight_compute_kernel_analysis/build --output-on-failure
```

The CUDA test runs every variant on dimensions with vectorized tail handling and compares every
element with the CPU reference. Python tests cover command construction, raw metric parsing,
correctness gates, and the rule that a kernel-only win is not automatically a deployment win. The
CUDA case is skipped when no device is accessible; actual profiler capture additionally requires
hardware-counter permission.

## Checkpoints

1. Use Nsight Systems to separate staging, transfers, NPP resize, custom conversion, and idle gaps.
2. Explain occupancy, registers, memory workload, scheduler activity, and warp stalls together;
   never optimize one metric in isolation.
3. Compare only matched variants and preserve Lesson 20's numerical error contract.
4. Report standalone kernel, complete GPU preprocessing, and matched pipeline evidence separately.
5. Reject a deployment-benefit claim when kernel metrics improve but preprocessing or end-to-end
   results do not; concluding that further optimization has low expected value is acceptable.
