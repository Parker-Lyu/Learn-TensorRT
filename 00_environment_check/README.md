# 00 - Environment Check

Goal: confirm that the GPU, CUDA, TensorRT, C++, and Python environment required by later lessons
is working.

The course uses a development container derived from:

```text
nvcr.io/nvidia/pytorch:25.11-py3
```

## Prepare The Development Environment

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

## Enter The Container

Open an independent shell with:

```bash
docker exec -it learn-tensorrt bash
```

The interactive main shell is also attachable:

```bash
docker attach learn-tensorrt
```

When using `docker attach`, press `Ctrl-p`, then `Ctrl-q` to detach without stopping the container.

## Check The Environment

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
ModelOpt, Ultralytics, ONNX, ONNX Runtime, and the ONNX simplification tools used by the course.
