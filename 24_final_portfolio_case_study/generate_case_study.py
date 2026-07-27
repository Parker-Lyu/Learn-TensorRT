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


def main() -> int:
    report10 = read(ROOT / "reports/10a_end_to_end_validation.md")
    report12 = read(ROOT / "reports/12a_precision_performance.md")
    report17 = read(ROOT / "reports/17a_pipeline_performance.md")
    checks = json.loads(read(ROOT / "24_final_portfolio_case_study/outputs/local_checks.json"))
    performance = section(report12, "Performance", "Detection Quality and Release Gate")
    quality = section(
        report12, "Detection Quality and Release Gate", "Raw Tensor Drift Versus TensorRT FP32"
    )
    fp32, fp16, int8 = (row(performance, name) for name in ("FP32", "FP16", "INT8"))
    gates = {name: row(quality, name)[-1] for name in ("FP32", "FP16", "INT8")}
    if gates["INT8"] == "PASS":
        precision_decision = "INT8 is the current deployment candidate under the declared gate."
        precision_next = "confirm INT8 behavior on the target deployment hardware"
    elif gates["FP16"] == "PASS":
        precision_decision = (
            "FP16 is the current deployment choice; INT8 remains blocked by the declared "
            "fixed-dataset accuracy gate."
        )
        precision_next = "improve INT8 calibration, mixed precision, or QAT"
    else:
        precision_decision = "FP32 remains the deployment baseline because reduced precision fails."
        precision_next = "investigate FP16/INT8 numerical drift"
    single = re.search(
        r"\| (\d+) \| (\d+) \| (\d+) \| (\d+) \| ([0-9.]+) \| "
        r"([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|",
        report17,
    )
    if not single:
        raise ValueError("missing single-stream evidence")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
    docker_sizes = ROOT / "24_final_portfolio_case_study/outputs/docker_image_sizes.txt"
    docker_note = (
        read(docker_sizes).strip()
        if docker_sizes.exists()
        else "Not generated; run `24_final_portfolio_case_study/measure_images.sh`."
    )
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

- [10a functional and architecture validation](10a_end_to_end_validation.md)
- [12a precision and performance decision](12a_precision_performance.md)
- [17a pipeline performance and reliability](17a_pipeline_performance.md)

The linked reports retain commands and raw-artifact paths; this case study does not duplicate them.

## Verification Environment

- Development image: `{platform_evidence['development_image']}`
- Container build identity: `{platform_evidence['container_build_id']}`
- Declared course GPU: `{platform_evidence['declared_gpu']}`
- Runtime GPU / compute capability / driver / memory MiB query:
  `{gpu['stdout'].strip() or gpu['stderr'].strip()}`
- TensorRT: `{tensorrt['stdout'].strip() or tensorrt['stderr'].strip()}`
- CUDA Toolkit query: `{cuda['stdout'].strip() or cuda['stderr'].strip()}`

Performance and acceptance results are valid only for their recorded hardware and software identity.

## Precision Decision

| Precision | Samples | Mean ms | P50 ms | P90 ms | P99 ms | Images/s | Accuracy gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP32 | {' | '.join(fp32)} | {gates['FP32']} |
| FP16 | {' | '.join(fp16)} | {gates['FP16']} |
| INT8 | {' | '.join(int8)} | {gates['INT8']} |

{precision_decision} The decision uses the canonical 5,000-image human-labeled validation split and
identity-linked engine evidence. A one-input Polygraphy alignment is valuable for locating numerical
conversion bugs, but it cannot replace dataset accuracy, tail latency, or long-run reliability.

## Pipeline Result

| Captured | Processed | Dropped | Queue peak | FPS | P50 ms | P90 ms | P99 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| {' | '.join(single.groups())} |

The latest-frame bounded queue trades completeness for freshness under sustained overload. Normal
EOS drains work; cancellation and failure discard queued work and join all workers. Multi-stream
round-robin scheduling protects fairness, while dynamic batching trades queue delay for throughput.

## Reusable Structure

- C++ preprocessing, postprocessing, TensorRT runner, and visualization libraries in lesson 10.
- Generic bounded queues and lifecycle tests in lessons 13–16.
- CUDA/NPP preprocessing with pageable, pinned, and mapped-memory evidence in lesson 17.
- Stable C ABI plus Python ctypes wrapper in lesson 21.
- Focused algorithm and RAII tests in lesson 23.

## Test Evidence

| Check | Result |
| --- | --- |
{test_rows}

Local-equivalent status: **{'PASS' if checks['passed'] else 'FAIL'}**. The JSON evidence retains each
command's stdout, stderr, return code, duration, and platform query. A failed or unavailable GPU check
remains failed rather than being converted into a CPU-only pass. The formal 30-minute soak and a
successful ThreadSanitizer run remain separate release gates.

## Delivery Image

`24_final_portfolio_case_study/Dockerfile` uses the pinned development image as its builder and a
CUDA 13.0 Ubuntu 24.04 base as its runtime stage. The runtime contains the executable, TensorRT 10
runtime library, OpenCV runtime packages, one engine, and `assets/img.jpeg`. Engines must be rebuilt
for the deployment environment.

Image-size evidence: `{docker_note}`

## Completed Elective Track

The advanced NVIDIA deployment path includes Triton configuration/client load tests, ONNX graph
surgery, a real custom TensorRT CUDA plugin, DeepStream multi-stream configuration/parser work,
Jetson/DLA target procedures, a C ABI Python integration, and LLM inference awareness. Triton,
DeepStream, and Jetson runtime acceptance remains explicitly hardware/container dependent.

## Bottleneck and Future Work

Nsight evidence identified CPU preprocessing/postprocessing as the original end-to-end bottleneck.
CUDA/NPP reduced preprocessing work, but transfer strategy still matters. Next work is to
{precision_next}, complete the formal soak/TSAN gates, and validate runtime behavior on Triton,
DeepStream, and Jetson hardware.

## Resume Bullets

- Built and validated a modular C++17 YOLOv8 TensorRT pipeline with FP32/FP16/INT8 release gates,
  CUDA/NPP preprocessing, dynamic batching, bounded multi-stream scheduling, and fault injection.
- Implemented a serialized CUDA TensorRT plugin, ONNX graph repair workflow, C ABI shared library,
  Python integration, and reproducible latency/accuracy/report generation.
- Profiled CPU/GPU bottlenecks and documented honest deployment boundaries for Triton, DeepStream,
  Jetson DLA, OpenVINO CPU, and local autoregressive LLM inference.

## Five-Minute English Presentation

Start with the deployment goal and controlled YOLOv8 model. Explain ONNX/TensorRT correctness and
why raw alignment precedes detection metrics. Present the measured precision decision from the 12a
gate. Walk through the bounded single/multi-stream design and capture-to-result tail latency. Show
the CUDA preprocessing and custom plugin evidence. Finish with reproducibility,
remaining soak/TSAN/hardware gates, and why those limitations are reported rather than hidden.

## Longer Interview Walkthrough

Discuss RAII ownership, dynamic profiles, tensor offsets, decode/NMS coordinate transforms, queue
close semantics, exception propagation, scheduling fairness, CUDA streams and transfer modes,
plugin registration/serialization, C ABI ownership, and deployment image contents. For every number,
trace it to the linked generated report and raw artifact; if evidence is incomplete, state the exact
command and environment needed to complete it.
"""
    output = ROOT / "reports/24_final_portfolio_case_study.md"
    output.write_text(text, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
