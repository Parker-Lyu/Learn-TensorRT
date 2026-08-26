#!/usr/bin/env python3
"""Generate the lesson 31 decision report from local profiler evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LESSON = Path(__file__).resolve().parent


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_benchmark(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported kernel benchmark schema")
    results = data.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("kernel benchmark requires non-empty results")
    for result in results:
        if result.get("maximum_absolute_error", 1.0) > 1e-6:
            raise ValueError(f"kernel correctness failed: {result.get('variant', 'unknown')}")
        timing = result.get("timing_ms", {})
        if not all(isinstance(timing.get(key), (int, float)) and timing[key] > 0
                   for key in ("mean", "p50", "p90")):
            raise ValueError("kernel timing summary is incomplete")
    return results


def kernel_decision(results: list[dict[str, Any]]) -> tuple[str, float]:
    by_name = {result["variant"]: result for result in results}
    if "baseline_16x16" not in by_name:
        raise ValueError("baseline_16x16 result is required")
    baseline = by_name["baseline_16x16"]["timing_ms"]["p50"]
    best = min(results, key=lambda item: item["timing_ms"]["p50"])
    improvement = (baseline - best["timing_ms"]["p50"]) / baseline * 100.0
    if best["variant"] == "baseline_16x16" or improvement <= 0.0:
        return (
            "No variant reduces the observed baseline P50; retain the readable baseline unless "
            "new system-level evidence justifies more work.",
            max(0.0, improvement),
        )
    return (
        f"{best['variant']} improves standalone kernel P50 by {improvement:.2f}%, but it is not a "
        "deployment win until matched GPU-preprocessing and pipeline evidence also improve.",
        improvement,
    )


def load_lesson20_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def lesson20_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "| Not captured | - | - | - | - | - | - | - |"
    rendered = []
    for row in rows:
        stage = float(row["host_stage_ms"])
        h2d = float(row["h2d_ms"])
        resize = float(row["npp_resize_ms"])
        conversion = float(row["conversion_ms"])
        gpu = float(row["gpu_preprocess_ms"])
        d2h = float(row["d2h_ms"])
        measured_path = stage + h2d + gpu + d2h
        conversion_share = conversion / measured_path * 100.0 if measured_path > 0.0 else 0.0
        rendered.append(
            f"| {row['mode']} | {stage:.4f} | {h2d:.4f} | {resize:.4f} | {conversion:.4f} | "
            f"{gpu:.4f} | {measured_path:.4f} | {conversion_share:.2f}% |"
        )
    return "\n".join(rendered)


def compact_metrics(metrics: dict[str, Any]) -> str:
    rows = []
    for variant, items in metrics.get("variants", {}).items():
        by_kernel: dict[str, list[dict[str, str]]] = {}
        for item in items:
            by_kernel.setdefault(item.get("kernel", "unknown"), []).append(item)
        for kernel, kernel_items in by_kernel.items():
            exact_metrics = {
                "launch__registers_per_thread",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                "l1tex__t_sector_hit_rate.pct",
                "lts__t_sector_hit_rate.pct",
            }
            selected = [item for item in kernel_items if item.get("metric") in exact_metrics]
            stalls = [
                item for item in kernel_items
                if item.get("metric", "").startswith("smsp__average_warps_issue_stalled_")
            ]
            if stalls:
                selected.append(max(stalls, key=lambda item: float(item.get("value", "0"))))
            short_kernel = kernel.split("(", 1)[0]
            for item in selected:
                rows.append(
                    f"| {variant} / `{short_kernel}` | `{item.get('metric', '')}` | "
                    f"{item.get('value', '')} {item.get('unit', '')} |"
                )
    return "\n".join(rows) if rows else "| Not captured | - | - |"


def pipeline_section(manifest: dict[str, Any]) -> str:
    pipeline = manifest.get("pipeline", {})
    if not pipeline.get("available"):
        return (
            "No matched Lesson 21 pipeline evidence was supplied. Kernel and preprocessing results "
            "therefore cannot be described as an end-to-end deployment improvement."
        )
    metrics = load_json(Path(pipeline["artifact"]))
    required = ("fps", "p50_ms", "p90_ms", "p99_ms", "preprocess_ms", "inference_ms")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise ValueError(f"Lesson 21 pipeline evidence is missing: {', '.join(missing)}")
    return (
        f"Matched Lesson 21 evidence reports `{metrics['fps']:.3f}` FPS, capture-to-dispatch "
        f"P50/P90/P99 `{metrics['p50_ms']:.3f}/{metrics['p90_ms']:.3f}/"
        f"{metrics['p99_ms']:.3f}` ms, cumulative GPU preprocessing "
        f"`{metrics['preprocess_ms']:.3f}` ms, and cumulative TensorRT inference "
        f"`{metrics['inference_ms']:.3f}` ms. These values are accepted only when its recorded "
        "environment and workload match the comparison."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=LESSON / "outputs")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "reports/31_nsight_compute_kernel_analysis.md"
    )
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    benchmark = load_json(input_dir / "kernel_benchmark.json")
    results = validate_benchmark(benchmark)
    manifest = load_json(input_dir / "profile_manifest.json")
    metrics = load_json(input_dir / "ncu_metrics_summary.json")
    lesson20_rows = load_lesson20_rows(input_dir / "lesson20_preprocess_benchmark.csv")
    decision, improvement = kernel_decision(results)

    timing_rows = "\n".join(
        f"| {item['variant']} | {item['kernel_launches_per_iteration']} | "
        f"{item['timing_ms']['mean']:.6f} | {item['timing_ms']['p50']:.6f} | "
        f"{item['timing_ms']['p90']:.6f} | {item['maximum_absolute_error']:.8f} |"
        for item in results
    )
    environment = manifest["environment"]
    nsys = manifest.get("nsight_systems", {})
    ncu = manifest.get("nsight_compute", {})
    text = f"""# Lesson 31 Nsight Compute Kernel Analysis

## Decision

{decision}

The best standalone improvement is `{improvement:.2f}%`. Nsight Compute replay duration is excluded
from all timing comparisons. A better occupancy, stall, or memory metric is diagnostic evidence, not
an optimization result by itself.

## Environment

- Course image: `{environment['course_image']}`
- Container release/build: `{environment['container_release']}` / `{environment['container_build_id']}`
- GPU, compute capability, driver, memory, clocks, P-state, power limit: `{environment['gpu_query']}`
- CUDA Toolkit: `{environment['cuda_toolkit']}`
- Nsight Systems: `{environment['nsight_systems'].splitlines()[-1]}`
- Nsight Compute: `{environment['nsight_compute'].splitlines()[-1]}`

## Standalone Kernel Timing

| Variant | Launches | Mean ms | P50 ms | P90 ms | Max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
{timing_rows}

All variants use the same deterministic 640x640 packed-BGR input, preallocated buffers, warmup,
iteration count, CUDA event boundaries, and exact CPU reference. The unfused row includes both
kernel launches and its intermediate global-memory traffic.

## Nsight Systems Evidence

Capture available: **{'yes' if nsys.get('available') else 'no'}**. The Lesson 20 timeline separates
host staging, H2D, NPP resize submission, conversion submission, D2H, and the corresponding CUDA
work across pageable, pinned, and mapped modes. The Lesson 20 table below quantifies conversion's
share of the measured path rather than declaring a hotspot from kernel duration alone. Mapped mode
executes against host-visible memory on this discrete GPU, so its conversion duration is evidence
about that transfer strategy, not a device-memory kernel baseline. If the custom conversion kernel
is not material in the target path, this lesson treats that as evidence to stop.

## Nsight Compute Metrics

Capture available: **{'yes' if ncu.get('available') else 'no'}**. Raw reports remain local and
environment-specific; the table is regenerated from the selected 2025.3.1 sections.

| Variant | Metric | Value |
| --- | --- | ---: |
{compact_metrics(metrics)}

Occupancy is not maximized as an objective. Warp stalls are interpreted together with scheduler,
memory-workload, register, and Speed-of-Light evidence. A bandwidth-limit conclusion requires
measured throughput relative to peak plus an explanation of bytes moved per output.

## Lesson 20 GPU Preprocessing

| Memory mode | Host stage ms | H2D ms | NPP resize ms | Conversion ms | GPU preprocess ms | Measured path ms | Conversion share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{lesson20_table(lesson20_rows)}

## End-to-End Pipeline

{pipeline_section(manifest)}

## Boundary

Kernel metrics improving without matched GPU-preprocessing and end-to-end improvement is not a
deployment gain. It is valid to retain the baseline or conclude that further optimization has low
expected value. TensorRT internal kernels are outside this lesson; Lesson 26 provides a later custom
plugin-kernel case.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
