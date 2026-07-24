# 00 - Environment Check

Goal: build a reproducible GPU development container and record its driver, CUDA, TensorRT,
compiler, Python, and OpenCV environment before running deployment experiments.

The host remains minimal: NVIDIA driver, Docker Engine, NVIDIA Container Toolkit, and the source
repository. CUDA, cuDNN, TensorRT, compilers, OpenCV, and Python course dependencies live in the
container derived from:

```text
nvcr.io/nvidia/pytorch:25.11-py3
```

This pins the course to CUDA Toolkit 13.0 and TensorRT 10.14.1.48. The repository's development
Dockerfile is `docker/Dockerfile.dev`; complete host checks, build instructions, persistent
container commands, attach instructions, and troubleshooting rules are in
[`agent_env_setup.md`](agent_env_setup.md).

## Build And Start

Run from the repository root:

```bash
docker build \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  -f docker/Dockerfile.dev \
  -t learn-tensorrt:25.11 .

docker run -dit \
  --name learn-tensorrt \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount type=bind,source="$(pwd)",target=/workspace/Learn-TensorRT \
  --workdir /workspace/Learn-TensorRT \
  learn-tensorrt:25.11
```

Open additional shells at any time with:

```bash
docker exec -it learn-tensorrt bash
```

`docker attach learn-tensorrt` is also supported. Use `Ctrl-p`, `Ctrl-q` to detach without stopping
the container. VS Code users can instead attach through `Dev Containers: Attach to Running
Container...`.

## Run The Check

Inside the container:

```bash
cd /workspace/Learn-TensorRT
bash 00_environment_check/check_env.sh
```

The script prints a concise report to the terminal. Save reproducibility evidence when needed:

```bash
bash 00_environment_check/check_env.sh |& tee 00_environment_check/env_report.md
```

`env_report.md` is machine-specific evidence; refresh it only when intentionally recording the
environment used for a lesson report.

## Acceptance Criteria

- The persistent `learn-tensorrt` container is running with GPU access.
- The bind-mounted project is writable without creating root-owned host files.
- `nvidia-smi`, CUDA 13.0 `nvcc`, and TensorRT 10.14 `trtexec` work.
- TensorRT C++ libraries and the TensorRT Python package are visible.
- NVIDIA PyTorch 25.11 and ModelOpt remain available.
- CMake, the C++17 compiler, and OpenCV C++ development files are installed.
- Python imports Ultralytics, ONNX, and ONNX Runtime.

Do not install or upgrade CUDA, TensorRT, PyTorch, or ModelOpt independently: those components come
from the pinned NVIDIA image and form one compatibility-tested stack.
