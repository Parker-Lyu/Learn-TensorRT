# 00 - Environment Check

## Purpose

- Record the local machine, driver, CUDA, TensorRT, Python, compiler, and OpenCV versions.
- Keep reproducible commands for checking the environment.
- Verify the single `nvcr.io/nvidia/pytorch:25.11-py3` development environment before later lessons.

## Prerequisites

- An NVIDIA GPU host with a compatible driver, Docker, and NVIDIA Container Toolkit.
- Access to the pinned `nvcr.io/nvidia/pytorch:25.11-py3` upstream image.

## Deliverables

- `check_env.sh` environment verifier
- `agent_env_setup.md` container-preparation guide
- `README.md` with container entry and verification commands

## Setup

Use a coding agent such as Codex or Claude Code to prepare the course environment. Ask the agent
to:

1. Read the repository-level [`AGENTS.md`](../AGENTS.md).
2. Read the environment-specific agent guide
   [`agent_env_setup.md`](agent_env_setup.md).
3. Inspect the existing host and container environment before making changes.
4. Create, reuse, or repair the development container as appropriate.
5. Run `00_environment_check/check_env.sh` and report the verification result.

For example, give the agent this instruction:

> Read `AGENTS.md` and `00_environment_check/agent_env_setup.md`, then inspect my current
> environment and prepare the development container for this course. Reuse working components
> instead of reinstalling them, and run the lesson 00 environment check when finished.

Some host setup operations may require administrator privileges. Review those operations and
provide authentication when prompted. Do not give the agent your password in chat.

### Enter The Container

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
