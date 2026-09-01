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

## Appendix: Jetson Quick Start

### The platform in one minute

Jetson is NVIDIA's family of embedded computers. Unlike a desktop GPU, a Jetson board combines
the CPU, GPU, memory controller, multimedia blocks, and (on supported generations) one or more
DLAs in a single power-constrained module. The exact hardware, power mode, and software release
are therefore part of an inference benchmark's identity.

| Term | Meaning | Why it matters in this lesson |
| --- | --- | --- |
| **JetPack SDK** | NVIDIA's versioned software stack and installation mechanism for Jetson. It bundles the board support package and libraries such as CUDA, TensorRT, cuDNN, VPI, and multimedia components. | Treat the JetPack release as one compatibility unit; do not independently replace its CUDA or TensorRT packages. |
| **L4T** | *Linux for Tegra*, the Jetson Linux base release: bootloader, Linux kernel, NVIDIA drivers, firmware, root filesystem, and system utilities. | The L4T release identifies the low-level platform on which the JetPack libraries run. |
| **CUDA** | NVIDIA's GPU programming platform and runtime. | TensorRT uses CUDA streams and device memory; the CUDA version is coupled to the JetPack release. |
| **TensorRT** | The inference optimizer and runtime that builds and executes serialized engines. | Engines must be built with the target JetPack/TensorRT/device combination. |
| **DLA** | *Deep Learning Accelerator*, a dedicated inference accelerator in selected Jetson SoCs. | Only a supported subset of TensorRT layers and precisions can run on DLA. Unsupported layers either make a DLA-only build fail or fall back to the GPU. |
| **GPU fallback** | TensorRT's option to execute layers unsupported by DLA on the GPU. | A DLA engine with fallback is a mixed GPU/DLA engine, not evidence that the complete network ran on DLA. |

The relationship is approximately: **JetPack release -> Jetson Linux/L4T + drivers -> CUDA,
TensorRT, cuDNN, and other SDK libraries**. Record all of these versions in the platform manifest
before comparing engines. A serialized engine is an environment-specific artifact, not a portable
model file.

### First checks on a Jetson

Run these commands on the target board (or in its JetPack-compatible container):

```bash
uname -m                         # expected: aarch64 on the board
cat /etc/nv_tegra_release        # L4T/Jetson Linux release
sudo -H tegrastats               # clocks, temperatures, power, and accelerator load
trtexec --version                # TensorRT version available to the process
python3 28_jetson_orin_xavier_dla_deployment/check_platform.py
```

`sudo` may request the Jetson user's password. `tegrastats` is a live monitor; stop it with
`Ctrl-C` after collecting the same warm-up and measurement interval used by the benchmark. The
`check_platform.py` output is a compatibility manifest, not proof that a model has executed on
DLA.

### Native versus cross compilation

- **Native build:** copy the source and ONNX model to the Jetson and build there. This is the
  simplest route and guarantees that TensorRT sees the target's libraries and hardware.
- **Cross compilation:** compile aarch64 code on x86 with a Jetson sysroot/toolchain, then run and
  build the TensorRT engine on the Jetson. The sysroot must match the target L4T release; an x86
  build alone cannot provide DLA execution evidence.
- **Containers:** a container can package user-space dependencies, but it still needs the Jetson
  host's compatible NVIDIA driver, device access, and JetPack boundary. A generic desktop CUDA
  container is not a substitute for a Jetson-compatible image.

### DLA expectations

DLA is an accelerator, not a second general-purpose GPU. Operators, tensor shapes, data types, and
memory limits determine whether TensorRT can place a layer on it. Start with FP16 or INT8 only when
the target and model support it, inspect the TensorRT layer-assignment log, and count GPU fallback
layers. Compare latency, throughput, power, and thermals at the same Jetson power mode and clock
configuration; DLA can be useful for system-level concurrency even when single-stream latency is
not lower.
