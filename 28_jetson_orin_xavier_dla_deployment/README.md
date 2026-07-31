# 28 - Jetson Orin/Xavier and DLA Deployment

## Purpose

This edge extension makes Jetson compatibility and DLA fallback evidence explicit. It can be
prepared on x86, but engine building and performance acceptance must run on the target Jetson.

## Prerequisites

- Full acceptance requires a Jetson Orin/Xavier target with a documented JetPack installation.
- x86 can run only the platform and tool tests; it cannot produce DLA execution evidence.

## Deliverables

- Platform check, engine-build, fallback-analysis, and benchmark tools
- Jetson-native build and DLA verification procedure
- CPU-only tool tests for x86 development

## Environment Boundary

JetPack couples the L4T kernel, CUDA, TensorRT, cuDNN, DeepStream, firmware, and NVIDIA drivers.
Do not install or upgrade CUDA/TensorRT independently with generic host packages. Start from a
documented JetPack image, record the exact target manifest, and build engines on that target.

TensorRT engines are not portable from the x86 RTX 4090 development environment to Jetson. An
engine built on one JetPack, TensorRT, and device combination must not be assumed valid on another.

Check the execution platform:

```bash
python3 28_jetson_orin_xavier_dla_deployment/check_platform.py
```

An x86 development machine reports `is_jetson=false` and zero DLA cores. That is a compatibility
check, not Jetson acceptance.

## DLA Trade-offs

- DLA can reserve GPU capacity for other CUDA workloads and may improve system-level power or
  concurrency even when its isolated latency is slower.
- GPU fallback introduces synchronization and transfers between device engines; count fallback
  layers before attributing performance to DLA.
- INT8 may improve DLA efficiency but requires a target-representative calibration set and the same
  accuracy gate used in lesson 14.
- A DLA-only build without fallback is a useful compatibility experiment: failure identifies layers
  that require model changes, precision changes, or GPU execution.

## Run

Copy the repository and ONNX model to the Jetson, enter the JetPack environment, then run:

```bash
./28_jetson_orin_xavier_dla_deployment/build_target_engines.sh
```

The script refuses non-Jetson systems. It builds one GPU FP16 engine and one DLA-core-0 FP16 engine
with explicit GPU fallback. YOLOv8 may contain layers that DLA cannot execute; fallback keeps the
model runnable but means the result is not a DLA-only claim.

Inspect fallback evidence and benchmark both target-local engines:

```bash
python3 28_jetson_orin_xavier_dla_deployment/analyze_fallback.py
python3 28_jetson_orin_xavier_dla_deployment/benchmark_target.py
```

`benchmark_target.py` embeds `outputs/platform_manifest.json` in its result. Use `tegrastats` in a
separate terminal to record clocks, temperatures, power, CPU/GPU/DLA load, and memory. Lock the same
power mode and clocks before comparisons, document warmup, and save the raw logs with the benchmark
JSON.

## Outputs

- Target-local engines, layer-assignment logs, platform manifest, benchmark JSON, and `tegrastats`
  captures belong under ignored `outputs/`.
- x86 output must be labeled as compatibility evidence, not DLA execution.

## Tests

```bash
python3 -m unittest discover -s 28_jetson_orin_xavier_dla_deployment/tests -v
python3 28_jetson_orin_xavier_dla_deployment/check_platform.py
```

Full acceptance remains pending until a Jetson target builds both engines, records layer assignment,
runs at least 100 measured samples per backend, captures `tegrastats`, and repeats detection-quality
validation. The x86 check must never be presented as DLA execution evidence.

## Checkpoints

1. Plan a reproducible Jetson-native or aarch64 cross-compiled TensorRT deployment.
2. Evaluate DLA compatibility, GPU fallback, power mode, clocks, thermals, and platform version coupling.
3. Record which validation is possible on x86 and which evidence requires target Jetson hardware.
