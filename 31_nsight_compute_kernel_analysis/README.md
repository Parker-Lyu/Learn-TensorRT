# 31 - Profiling-Driven CUDA Kernel Optimization

## Purpose

Practice the production optimization order on a complete, manually assembled inference workload:
use Nsight Systems to find material GPU work, use Nsight Compute to explain a source-owned kernel,
make one evidence-backed algorithmic change, and require both operator and complete-network A/B
timing before accepting it.

The workload is a small FP32 `Linear -> LayerNorm -> Linear` MLP. cuBLAS implements the linear
layers. The lesson owns the LayerNorm CUDA code, making it possible to move from system diagnosis to
source-level optimization without pretending that TensorRT or cuBLAS internal kernels are editable.

## Prerequisites

- Complete Lesson 13 so Nsight Systems timelines, NVTX ranges, warmup, and synchronization
  boundaries are familiar.
- Understand the CUDA-kernel integration boundary introduced by Lesson 26. This lesson reuses that
  engineering pattern, not Lesson 26's `[1, 4]` demonstration model.
- Use the persistent pinned development container with an accessible NVIDIA GPU.


## Deliverables

- A C++17/CUDA two-layer MLP with cuBLAS linear layers and a CPU numerical reference
- A conventional two-launch LayerNorm baseline and a one-launch fused implementation
- Runtime launch selection through the CUDA occupancy API, without architecture-specific tuning
  tables, launch-configuration search, or JIT auto-tuning
- Separate CUDA-event distributions for LayerNorm and the complete network
- Staged Nsight Systems and Nsight Compute capture workflows
- An environment-specific decision report that can accept, reject, or stop the optimization

## Design

For each row, LayerNorm computes:

```text
y = (x - mean(x)) / sqrt(variance(x) + epsilon) * gamma + beta
```

The baseline is not an intentionally broken serial kernel. It uses a block reduction to calculate
mean and inverse standard deviation, stores those two values per row, then launches a grid-stride
kernel for normalization and affine transformation. This is a reasonable staged implementation.

The optimized kernel keeps the block reduction but performs normalization in the same row-wise
launch. It removes one kernel launch and the global-memory round trip for row statistics. That is the
only algorithmic variable. Input, weights, precision, epsilon, allocation boundaries, and cuBLAS
operations remain matched.

`cudaOccupancyMaxPotentialBlockSize` supplies a runtime upper bound, which the row width limits to
the number of whole warps that can do useful work. Occupancy is a portable starting heuristic, not
proof of optimality. This lesson deliberately excludes per-architecture configuration tables,
offline block-size searches, and JIT auto-tuning.

## Build

Run from the repository root inside the persistent development container:

```bash
cmake -S 31_nsight_compute_kernel_analysis \
  -B 31_nsight_compute_kernel_analysis/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build 31_nsight_compute_kernel_analysis/build --parallel
```

## Run

Run the matched CUDA-event benchmark; profiler duration is never used as benchmark evidence.

```bash
./31_nsight_compute_kernel_analysis/build/lesson31_mlp_benchmark \
  --variant all --warmup 20 --iterations 100 \
  --output 31_nsight_compute_kernel_analysis/outputs/mlp_benchmark.json
```

Example output (values vary slightly with GPU clocks):

```text
baseline layernorm_p50_ms=0.006848 network_p50_ms=0.016384 max_error=0.000000 reduction_block=128
fused layernorm_p50_ms=0.004384 network_p50_ms=0.014336 max_error=0.000000 reduction_block=128
wrote "31_nsight_compute_kernel_analysis/outputs/mlp_benchmark.json"
```

Capture the complete MLP with Nsight Systems first, so kernel selection is evidence-driven:

```bash
python3 31_nsight_compute_kernel_analysis/profile_kernels.py --skip-ncu
```

Example output:

```text
{
  "benchmark": "/workspace/Learn-TensorRT/31_nsight_compute_kernel_analysis/outputs/mlp_benchmark.json",
  "manifest": "/workspace/Learn-TensorRT/31_nsight_compute_kernel_analysis/outputs/profile_manifest.json",
  "metrics": "/workspace/Learn-TensorRT/31_nsight_compute_kernel_analysis/outputs/ncu_metrics_summary.json"
}
```

Open `outputs/nsys/mlp_inference.nsys-rep` or inspect
`outputs/nsys/nsys_stats.txt`. Use the CUDA kernel summary and the `network_*`, `linear_*`, and
`layernorm_*` NVTX GPU projections to answer:

1. Which kernels dominate the complete network?
2. Is LayerNorm material enough to justify source-level work?
3. Are the two baseline LayerNorm launches and their boundaries visible?

If LayerNorm is immaterial, stop and record that decision. If it is material, collect
microarchitectural evidence for only the source-owned LayerNorm kernels:

```bash
python3 31_nsight_compute_kernel_analysis/profile_kernels.py --skip-nsys
```

Example output:

```text
{
  "benchmark": "/workspace/Learn-TensorRT/31_nsight_compute_kernel_analysis/outputs/mlp_benchmark.json",
  "manifest": "/workspace/Learn-TensorRT/31_nsight_compute_kernel_analysis/outputs/profile_manifest.json",
  "metrics": "/workspace/Learn-TensorRT/31_nsight_compute_kernel_analysis/outputs/ncu_metrics_summary.json"
}
```

The second command preserves the existing Nsight Systems manifest. It profiles the baseline
statistics/apply kernels and the fused kernel with the same sections. Read launch count and block
size first, then registers, achieved occupancy, DRAM/cache behavior, scheduler activity, and warp
stalls together. Do not optimize an isolated counter.

After the investigation, collect both profilers in one non-interactive invocation (may take several
minutes):

```bash
python3 31_nsight_compute_kernel_analysis/profile_kernels.py
```

Example output:

```text
{
  "benchmark": "/workspace/Learn-TensorRT/31_nsight_compute_kernel_analysis/outputs/mlp_benchmark.json",
  "manifest": "/workspace/Learn-TensorRT/31_nsight_compute_kernel_analysis/outputs/profile_manifest.json",
  "metrics": "/workspace/Learn-TensorRT/31_nsight_compute_kernel_analysis/outputs/ncu_metrics_summary.json"
}
```

Generate the local decision report from the captured evidence:

```bash
python3 31_nsight_compute_kernel_analysis/generate_report.py
```

Example output:

```text
wrote /workspace/Learn-TensorRT/31_nsight_compute_kernel_analysis/outputs/optimization_decision.md
```

When profiler permission requires container root, run the profiling commands with
`docker exec --user root` from the host and restore output ownership using the actual configured
host UID/GID.

## Outputs

Committed deliverables are the C++/CUDA implementation, focused tests, profiling/report tools, and
this README. Generated artifacts remain ignored:

- `outputs/mlp_benchmark.json`: matched unprofiled operator and network timings
- `outputs/profile_manifest.json`: commands, measurement policy, tool versions, and GPU identity
- `outputs/nsys/mlp_inference.nsys-rep`: complete-workload timeline
- `outputs/nsys/nsys_stats.txt`: text summaries used for initial kernel selection
- `outputs/ncu/*.ncu-rep`: baseline and fused LayerNorm counter captures
- `outputs/ncu_metrics_summary.json`: compact metrics extracted from raw reports
- `outputs/optimization_decision.md`: local optimization decision

The workload is representative, not a deployed service. Results from one GPU or shape are not a
portable production claim.

## Tests

```bash
ctest --test-dir 31_nsight_compute_kernel_analysis/build --output-on-failure
```

The CUDA test exercises non-warp-aligned feature counts, both LayerNorm paths, complete-network CPU
comparison, runtime launch selection, invalid variants, and invalid dimensions. Python tests cover
profiler command boundaries, Nsight Compute CSV parsing, correctness gates, and the rule that an
operator-only win is insufficient. CUDA tests skip when no device is accessible; profiler capture
additionally requires the corresponding Nsight tool and hardware-counter permission.

## Failure Semantics

- Reject either implementation when maximum absolute error exceeds `2e-4` for the deterministic
  FP32 reference workload.
- Reject fusion when matched LayerNorm P50 does not improve.
- Do not accept fusion for this workload when LayerNorm improves but complete-network P50 does not.
- Stop before Nsight Compute when Nsight Systems shows the source-owned work is immaterial.
- Never use Nsight Compute replay duration as performance evidence.

## Checkpoints

1. Use the Systems evidence, rather than source familiarity, to justify selecting LayerNorm.
2. Explain why the baseline is a plausible implementation rather than an artificial one-thread
   strawman.
3. Connect the removed launch and row-statistics traffic to the relevant Compute metrics.
4. Explain why higher occupancy alone cannot accept the fused implementation.
5. Compare LayerNorm and complete-network P50 and make an explicit ship, reject, or stop decision.
6. State what real model shapes and service-level A/B evidence would be required before applying the
   conclusion to production.
