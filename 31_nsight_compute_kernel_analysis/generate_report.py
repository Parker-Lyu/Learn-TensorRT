#!/usr/bin/env python3
"""Generate the Lesson 31 optimization decision from local evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LESSON = Path(__file__).resolve().parent
CORRECTNESS_LIMIT = 2.0e-4


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_benchmark(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict) or data.get("schema_version") != 2:
        raise ValueError("unsupported MLP benchmark schema")
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("MLP benchmark requires non-empty results")
    by_name = {result.get("variant"): result for result in results}
    missing = {"baseline", "fused"} - by_name.keys()
    if missing:
        raise ValueError(f"MLP benchmark is missing variants: {', '.join(sorted(missing))}")
    for result in results:
        if result.get("maximum_absolute_error", 1.0) > CORRECTNESS_LIMIT:
            raise ValueError(f"network correctness failed: {result.get('variant', 'unknown')}")
        for timing_name in ("layernorm_timing_ms", "network_timing_ms"):
            timing = result.get(timing_name, {})
            if not all(isinstance(timing.get(key), (int, float)) and timing[key] > 0
                       for key in ("mean", "p50", "p90")):
                raise ValueError(f"{timing_name} is incomplete")
    return results


def optimization_decision(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {result["variant"]: result for result in results}
    baseline = by_name["baseline"]
    fused = by_name["fused"]
    baseline_operator = baseline["layernorm_timing_ms"]["p50"]
    fused_operator = fused["layernorm_timing_ms"]["p50"]
    baseline_network = baseline["network_timing_ms"]["p50"]
    fused_network = fused["network_timing_ms"]["p50"]
    operator_reduction = (baseline_operator - fused_operator) / baseline_operator * 100.0
    network_reduction = (baseline_network - fused_network) / baseline_network * 100.0
    if operator_reduction <= 0.0:
        conclusion = "The fused kernel is rejected because it does not improve matched LayerNorm P50."
    elif network_reduction <= 0.0:
        conclusion = (
            "The kernel optimization succeeds in isolation but is rejected for this workload because "
            "matched complete-network P50 does not improve."
        )
    else:
        conclusion = (
            "The fused implementation is accepted for this workload: it passes the numerical contract "
            "and improves both matched LayerNorm and complete-network P50."
        )
    return {
        "operator_reduction": operator_reduction,
        "operator_speedup": baseline_operator / fused_operator,
        "network_reduction": network_reduction,
        "network_speedup": baseline_network / fused_network,
        "operator_share": baseline_operator / baseline_network * 100.0,
        "conclusion": conclusion,
    }


def compact_metrics(metrics: dict[str, Any]) -> str:
    rows = []
    exact_metrics = {
        "launch__block_size", "launch__registers_per_thread",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__t_sector_hit_rate.pct", "lts__t_sector_hit_rate.pct",
    }
    for variant, items in metrics.get("variants", {}).items():
        by_kernel: dict[str, list[dict[str, str]]] = {}
        for item in items:
            by_kernel.setdefault(item.get("kernel", "unknown"), []).append(item)
        for kernel, kernel_items in by_kernel.items():
            selected = [item for item in kernel_items if item.get("metric") in exact_metrics]
            stalls = [
                item for item in kernel_items
                if item.get("metric", "").startswith("smsp__average_warps_issue_stalled_")
            ]
            numeric_stalls = []
            for item in stalls:
                try:
                    numeric_stalls.append((float(item.get("value", "0").replace(",", "")), item))
                except ValueError:
                    continue
            if numeric_stalls:
                selected.append(max(numeric_stalls, key=lambda pair: pair[0])[1])
            short_kernel = kernel.split("(", 1)[0]
            for item in selected:
                rows.append(
                    f"| {variant} / `{short_kernel}` | `{item.get('metric', '')}` | "
                    f"{item.get('value', '')} {item.get('unit', '')} |"
                )
    return "\n".join(rows) if rows else "| Not captured | - | - |"


def profiler_status(evidence: dict[str, Any], label: str) -> str:
    if evidence.get("available"):
        return f"{label} capture is available in the local output directory."
    return f"{label} capture is unavailable: {evidence.get('reason', 'no reason recorded')}."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=LESSON / "outputs")
    parser.add_argument(
        "--output", type=Path, default=LESSON / "outputs/optimization_decision.md"
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    benchmark = load_json(input_dir / "mlp_benchmark.json")
    results = validate_benchmark(benchmark)
    manifest = load_json(input_dir / "profile_manifest.json")
    metrics = load_json(input_dir / "ncu_metrics_summary.json")
    decision = optimization_decision(results)
    configuration = benchmark["configuration"]
    environment = manifest["environment"]
    nsys = manifest.get("nsight_systems", {})
    ncu = manifest.get("nsight_compute", {})

    timing_rows = "\n".join(
        f"| {item['variant']} | {item['layernorm_launches']} | "
        f"{item['launch_configuration']['reduction_block_size']} | "
        f"{item['layernorm_timing_ms']['p50']:.6f} | "
        f"{item['network_timing_ms']['p50']:.6f} | "
        f"{item['maximum_absolute_error']:.8f} |"
        for item in results
    )
    text = f"""# Lesson 31 MLP LayerNorm Optimization Report

## Decision

{decision['conclusion']}

- LayerNorm P50 reduction: `{decision['operator_reduction']:.2f}%` (`{decision['operator_speedup']:.2f}x`)
- Complete-network P50 reduction: `{decision['network_reduction']:.2f}%` (`{decision['network_speedup']:.2f}x`)
- Separately measured baseline LayerNorm/network ratio: `{decision['operator_share']:.2f}%`

The ratio is a prioritization signal, not an additive timing decomposition. LayerNorm and the full
network are measured in separate CUDA-event loops. The conclusion applies to this workload and GPU;
it is not a claim about an unrelated production model.

## Workload

The manually assembled network is `Linear -> LayerNorm -> Linear`. Linear layers use cuBLAS; only
the source-owned LayerNorm is changed. Shape: rows `{configuration['rows']}`, input features
`{configuration['input_features']}`, hidden features `{configuration['hidden_features']}`, output
features `{configuration['output_features']}`. Both variants use identical tensors, FP32 math,
epsilon, warmup, iteration count, allocation boundary, and CPU reference.

| Variant | LayerNorm launches | Runtime-selected reduction block | LayerNorm P50 ms | Network P50 ms | Max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
{timing_rows}

The baseline uses a row-statistics kernel followed by a normalization/affine kernel. The fused
variant computes statistics and writes normalized output in one row-wise kernel, eliminating the
mean/inverse-standard-deviation round trip and one launch. CUDA's occupancy API supplies a runtime
upper bound and row width prevents idle whole warps; the lesson performs no architecture-specific
table lookup, search, or JIT tuning.

## Nsight Systems

{profiler_status(nsys, 'Nsight Systems')} Read its CUDA kernel summary and NVTX GPU projection first.
The `network_*`, `linear_*`, and `layernorm_*` ranges establish whether a source-owned LayerNorm is
material enough to justify deeper analysis. Do not choose it merely because its source is available.

## Nsight Compute

{profiler_status(ncu, 'Nsight Compute')} When capture is permitted, the filter collects only
`layer_norm_*_kernel` launches. Profiler replay duration is excluded from all timing decisions.

| Variant / kernel | Metric | Value |
| --- | --- | ---: |
{compact_metrics(metrics)}

Interpret registers, achieved occupancy, DRAM throughput, cache hit rates, scheduler activity, and
dominant warp stalls together. Fusion is supported when the removed launch and intermediate traffic
explain measured timing without introducing a countervailing occupancy or stall regression.

## Environment

- Course image: `{environment['course_image']}`
- Container release/build: `{environment['container_release']}` / `{environment['container_build_id']}`
- GPU, compute capability, driver, memory, clocks, P-state, power limit: `{environment['gpu_query']}`
- CUDA Toolkit: `{environment['cuda_toolkit']}`
- Nsight Systems: `{environment['nsight_systems'].splitlines()[-1]}`
- Nsight Compute: `{environment['nsight_compute'].splitlines()[-1]}`

## Production Boundary

This is a representative inference workload, not evidence from a deployed service. Shipping the
same change elsewhere requires a matched A/B in that model and its real request shapes. If Nsight
Systems shows LayerNorm has negligible share, or complete-network timing does not improve, stop even
when an isolated kernel counter looks better.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
