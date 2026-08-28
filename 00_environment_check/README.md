# 00 - Environment Check

## Purpose

Build and verify the single development environment used throughout this course. Confirm that the
pinned container provides GPU access and all required CUDA, TensorRT, C++ and Python tooling before
starting later lessons.

## Prerequisites

- An NVIDIA GPU host with a compatible driver, Docker, and NVIDIA Container Toolkit.

## Deliverables

- `check_env.sh` environment verifier
- `agent_env_setup.md` container-preparation guide
- `README.md` with container entry and verification commands

## Setup

Use a coding agent such as Codex or Claude Code to prepare the course environment. Ask the agent
to:

> Read `AGENTS.md` and `00_environment_check/agent_env_setup.md`, then inspect my current
> environment and prepare the development container for this course. Reuse working components
> instead of reinstalling them, and run the lesson 00 environment check when finished.

Some host setup operations may require administrator privileges. Review those operations and
provide authentication when prompted. Do not give the agent your password in chat.

### How To Enter The Container

Open an independent shell with:

```bash
docker exec -it learn-tensorrt bash
```

The interactive main shell is also attachable:

```bash
docker attach learn-tensorrt
```

When using `docker attach`, press `Ctrl-p`, then `Ctrl-q` to detach without stopping the container.  

**All subsequent code for this course will run inside this container.**

## Run

Inside the container, run:

```bash
cd /workspace/Learn-TensorRT
bash 00_environment_check/check_env.sh
```

The base development environment is ready when the script ends with:

```text
[PASS] Environment checks passed.
```

The script checks the GPU, CUDA 13.0, TensorRT 10.14, CMake, the C++ compiler, OpenCV, PyTorch,
ModelOpt and its ONNX AutoCast dependencies, Ultralytics, ONNX, ONNX Runtime, and the ONNX
simplification tools used by the course.

ONNX Runtime is intentionally the CPU `onnxruntime==1.22.0` distribution in the baseline image.
It is used for deterministic ONNX and ModelOpt AutoCast numerical validation, while TensorRT is the
GPU deployment runtime taught by the course. ModelOpt 0.37's Linux extra requests
`onnxruntime-gpu`, but that optional CUDA Execution Provider workflow is outside the baseline and
must be validated separately rather than mixed into this environment.

## Outputs

- The main output is the environment-check log written to standard output.
- A successful run ends with `[PASS] Environment checks passed.`; redirect the log when it will be
  used as report evidence.

## Checkpoints

1. Verify that the pinned development container exposes the required GPU, CUDA, TensorRT, compiler,
  Python, and course dependencies.
2. Explain the compatibility boundary between the host driver and the container-provided CUDA and TensorRT stack.
