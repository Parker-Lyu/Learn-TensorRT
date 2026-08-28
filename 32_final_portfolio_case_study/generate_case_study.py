#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def section(report: str, heading: str, next_heading: str) -> str:
    match = re.search(rf"## {re.escape(heading)}\n(.*?)\n## {re.escape(next_heading)}", report, re.S)
    if not match:
        raise ValueError(f"missing {heading} section")
    return match.group(1)


def row(report_section: str, precision: str) -> list[str]:
    match = re.search(rf"\| {precision} \| ([^\n]+)", report_section)
    if not match:
        raise ValueError(f"missing {precision} performance row")
    return [value.strip() for value in match.group(1).split("|") if value.strip()]


def markdown_table(report: str, heading: str) -> list[dict[str, str]]:
    heading_match = re.search(rf"^## {re.escape(heading)}\s*$", report, re.M)
    if not heading_match:
        raise ValueError(f"missing {heading} section")

    section_text = report[heading_match.end() :]
    next_heading = re.search(r"^## ", section_text, re.M)
    if next_heading:
        section_text = section_text[: next_heading.start()]
    lines = section_text.splitlines()
    table_start = next(
        (index for index, line in enumerate(lines) if line.startswith("|")), None
    )
    if table_start is None or table_start + 2 >= len(lines):
        raise ValueError(f"missing table in {heading} section")

    def cells(line: str) -> list[str]:
        return [value.strip() for value in line.strip().strip("|").split("|")]

    headers = cells(lines[table_start])
    separator = cells(lines[table_start + 1])
    if len(separator) != len(headers) or not all(
        re.fullmatch(r":?-{3,}:?", value) for value in separator
    ):
        raise ValueError(f"malformed table in {heading} section")

    rows = []
    for line in lines[table_start + 2 :]:
        if not line.startswith("|"):
            break
        values = cells(line)
        if len(values) != len(headers):
            raise ValueError(f"malformed row in {heading} section")
        rows.append(dict(zip(headers, values)))
    if not rows:
        raise ValueError(f"empty table in {heading} section")
    return rows


def markdown_rows(rows: list[dict[str, str]], columns: tuple[str, ...]) -> str:
    try:
        return "\n".join(
            "| " + " | ".join(item[column] for column in columns) + " |"
            for item in rows
        )
    except KeyError as error:
        raise ValueError(f"missing table column: {error.args[0]}") from error


def docker_image_rows(path: Path) -> str:
    if not path.exists():
        return "| Not generated | - | - |"

    rows = []
    for line in read(path).splitlines():
        tags_text, image_id, size_text = line.rsplit(" ", 2)
        tags = ", ".join(json.loads(tags_text))
        size_mib = int(size_text) / (1024 * 1024)
        rows.append(
            f"| `{tags}` | `{image_id.removeprefix('sha256:')[:12]}` | {size_mib:.1f} |"
        )
    return "\n".join(rows)


def choose_precision(
    gates: dict[str, str], fp16_throughput: float, int8_throughput: float
) -> tuple[str, str]:
    if gates["INT8"] == "PASS" and int8_throughput > fp16_throughput:
        return (
            "INT8 is the current deployment choice because it passes the declared quality gate "
            "and outperforms matched FP16.",
            "confirm INT8 behavior on the target deployment hardware",
        )
    if gates["FP16"] == "PASS":
        if gates["INT8"] == "PASS":
            return (
                "FP16 is the current deployment choice: INT8 passes the declared quality gate "
                "but is slower than matched FP16.",
                "investigate the Q/DQ INT8 performance regression",
            )
        return (
            "FP16 is the current deployment choice; INT8 remains blocked by the declared fixed-"
            "dataset accuracy gate.",
            "improve INT8 calibration, mixed precision, or QAT",
        )
    return (
        "FP32 remains the deployment baseline because reduced precision fails.",
        "investigate FP16/INT8 numerical drift",
    )


def main() -> int:
    report12 = read(ROOT / "reports/12_end_to_end_validation.md")
    report15 = read(ROOT / "reports/15_precision_performance.md")
    report22 = read(ROOT / "reports/22_pipeline_performance.md")
    checks = json.loads(read(ROOT / "32_final_portfolio_case_study/outputs/local_checks.json"))
    performance = section(report15, "Performance", "Detection Quality and Release Gate")
    quality = section(
        report15, "Detection Quality and Release Gate", "Raw Tensor Drift Versus TensorRT FP32"
    )
    fp32, fp16, int8 = (row(performance, name) for name in ("FP32", "FP16", "INT8"))
    gates = {name: row(quality, name)[-1] for name in ("FP32", "FP16", "INT8")}
    fp16_throughput = float(fp16[-1])
    int8_throughput = float(int8[-1])
    precision_decision, precision_next = choose_precision(
        gates, fp16_throughput, int8_throughput
    )
    load_rows = markdown_table(report22, "Real Integrated Load Matrix")
    batch_one = next((item for item in load_rows if item.get("Batch") == "1"), None)
    if batch_one is None:
        raise ValueError("missing batch-1 integrated load evidence")
    policy_rows = markdown_table(report22, "Overload and Freshness Policies")
    batch_columns = ("Batch", "Completed", "FPS", "P50 ms", "P90 ms", "P99 ms", "Queue peak")
    policy_columns = (
        "Policy", "Captured", "Completed", "Evicted", "Aborted", "Queue peak", "FPS", "P99 ms"
    )
    batch_row = markdown_rows([batch_one], batch_columns)
    rendered_policy_rows = markdown_rows(policy_rows, policy_columns)
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
    docker_sizes = ROOT / "32_final_portfolio_case_study/outputs/docker_image_sizes.txt"
    docker_rows = docker_image_rows(docker_sizes)

    def check_status(item: dict) -> str:
        if item["returncode"] != 0:
            return "FAIL"
        if "Skipped" in item["stdout"]:
            return "PASS WITH SKIP"
        return "PASS"

    test_rows = "\n".join(
        f"| {item['name']} | {check_status(item)} |" for item in checks["checks"]
    )
    platform_evidence = checks["platform"]
    gpu = platform_evidence["gpu"]
    tensorrt = platform_evidence["tensorrt"]
    cuda = platform_evidence["cuda_toolkit"]
    text = f"""# Final TensorRT Deployment Portfolio Case Study

Repository evidence revision: `{commit}`. This project develops a YOLOv8n deployment from ONNX
export through TensorRT C++ inference, precision validation, bounded asynchronous pipelines, CUDA
preprocessing, server/edge integration exercises, and reusable language bindings.

## Evidence Map

- [12 functional and architecture validation](12_end_to_end_validation.md)
- [15 precision and performance decision](15_precision_performance.md)
- [22 pipeline performance and reliability](22_pipeline_performance.md)

The linked reports retain commands and raw-artifact paths; this case study does not duplicate them.

## Verification Environment

- Development image: `{platform_evidence['development_image']}`
- Container build identity: `{platform_evidence['container_build_id']}`
- Declared course GPU: `{platform_evidence['declared_gpu']}`
- Runtime GPU / compute capability / driver / memory MiB query:
  `{gpu['stdout'].strip() or gpu['stderr'].strip()}`
- TensorRT: `{tensorrt['stdout'].strip() or tensorrt['stderr'].strip()}`
- CUDA Toolkit: `{(cuda['stdout'].strip() or cuda['stderr'].strip()).splitlines()[-1]}`

Performance and acceptance results are valid only for their recorded hardware and software identity.

## Precision Decision

| Precision | Samples | Mean ms | P50 ms | P90 ms | P99 ms | Images/s | Accuracy gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP32 | {' | '.join(fp32)} | {gates['FP32']} |
| FP16 | {' | '.join(fp16)} | {gates['FP16']} |
| INT8 | {' | '.join(int8)} | {gates['INT8']} |

{precision_decision} The decision uses the canonical 5,000-image human-labeled validation split
and identity-linked engine evidence. A one-input Polygraphy alignment is valuable for locating
numerical conversion bugs, but it cannot replace dataset accuracy, tail latency, or long-run
reliability.

## Pipeline Result

| Batch | Completed | FPS | P50 ms | P90 ms | P99 ms | Queue peak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{batch_row}

| Policy | Captured | Completed | Evicted | Aborted | Queue peak | FPS | P99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rendered_policy_rows}

The bounded queue's policy determines whether overload applies producer backpressure or explicitly
evicts stale work. Normal EOS drains work; cancellation and failure discard queued work and join
all workers. Multi-stream round-robin scheduling protects fairness, while dynamic batching trades
queue delay for throughput.

## Reusable Structure

- C++ preprocessing, postprocessing, TensorRT runner, and visualization libraries in lesson 11.
- Generic bounded queues and lifecycle tests in lessons 16–19.
- CUDA/NPP preprocessing with pageable, pinned, and mapped-memory evidence in lesson 20.
- Stable C ABI plus Python ctypes wrapper in lesson 29.
- Reproducible CUDA kernel variants, correctness gates, and Nsight Compute analysis in lesson 31.

## Test Evidence

| Check | Result |
| --- | --- |
{test_rows}

Local-equivalent status: **{'PASS' if checks['passed'] else 'FAIL'}**. The JSON evidence retains each
command's stdout, stderr, return code, duration, and platform query. A failed or unavailable GPU check
remains failed rather than being converted into a CPU-only pass. The formal 30-minute soak and a
successful ThreadSanitizer run remain separate release gates.

## Delivery Image

`32_final_portfolio_case_study/Dockerfile` uses the pinned development image as its builder and a
CUDA 13.0 Ubuntu 24.04 base as its runtime stage. The runtime contains the executable, TensorRT 10
runtime library, OpenCV runtime packages, one engine, and `assets/img.jpeg`. Engines must be rebuilt
for the deployment environment.

| Image | ID | Size MiB |
| --- | --- | ---: |
{docker_rows}

## Elective Track Status

The implemented advanced NVIDIA exercises include Triton configuration and a client load-test tool,
ONNX graph surgery, a runnable custom TensorRT CUDA plugin, DeepStream multi-stream configuration
and parser code, Jetson/DLA target procedures, a C ABI Python integration, and LLM inference
awareness. This is not a fully completed runtime-validated elective track: Triton, DeepStream, and
Jetson acceptance remains pending in their required containers or target hardware.
"""
    output = ROOT / "reports/32_final_portfolio_case_study.md"
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
