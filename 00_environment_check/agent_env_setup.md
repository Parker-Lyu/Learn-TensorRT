# Agent Guide - Course Development Container

This document is the reproducible setup procedure for an AI coding agent preparing or verifying
this repository's development environment. The project uses a persistent GPU-enabled Docker
container. Do not install CUDA, cuDNN, TensorRT, OpenCV, or compiler toolchains directly on the
host.

## Pinned Baseline

The single upstream image is:

```text
nvcr.io/nvidia/pytorch:25.11-py3
```

The image supplies the course's CUDA, TensorRT, PyTorch, and ModelOpt stack. Do not replace or
upgrade those components in the course image:

- Ubuntu 24.04
- CUDA Toolkit 13.0
- TensorRT 10.14.1.48
- NVIDIA PyTorch 25.11 build
- NVIDIA ModelOpt 0.37.0

The repository Dockerfile adds C++ development tools, OpenCV C++ development files, Ultralytics,
ONNX, and ONNX Runtime. TensorRT engines and performance evidence remain specific to this pinned
environment and must be regenerated here.

Ubuntu's complete OpenCV development package transitively installs its UCX runtime through
VTK/OpenMPI, while NVIDIA PyTorch is linked to the newer HPC-X UCX bundled in the upstream image.
The Dockerfile sets `LD_PRELOAD` to the matching HPC-X `libucs`, `libucm`, and `libucp` libraries so
the two UCX ABIs are not mixed during `import torch`. Do not remove that setting without rerunning
the PyTorch import and CUDA checks.

## Host Responsibilities

The host needs only:

- a compatible NVIDIA driver (the CUDA 13.0 image requires driver 580.95.05 or newer on x86_64);
- Docker Engine with the Buildx and Compose plugins;
- NVIDIA Container Toolkit configured for Docker;
- this repository as a persistent bind mount.

Verify an already configured host before changing it:

```bash
nvidia-smi
docker version
docker info --format '{{json .Runtimes}}' | grep nvidia
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 nvidia-smi
```

If all four commands pass, do not reinstall the driver, Docker, or NVIDIA Container Toolkit.

### Host installation reference

Run this section only when the corresponding host component is missing. These commands must not be
run inside a container.

Install Docker Engine from Docker's official Ubuntu repository:

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
  sudo apt-get remove -y "$pkg"
done

sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
```

Log out and back in after changing Docker group membership (or use `newgrp docker`). Then install
and configure NVIDIA Container Toolkit if the `nvidia` runtime is missing:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## Build The Course Image

Run from the repository root. The UID/GID arguments ensure files written through the bind mount are
owned by the host user:

```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3
docker build \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  -f docker/Dockerfile.dev \
  -t learn-tensorrt:25.11 .
```

The build intentionally fails if Python dependency resolution replaces NVIDIA's PyTorch 25.11
build. Optional tools such as `onnxslim` and `onnxsim` are not part of the baseline; install them
only for a documented simplification experiment.

## Create The Persistent Container

From the repository root:

```bash
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

The container is deliberately persistent and interactive. It can be reused after a reboot:

```bash
docker start learn-tensorrt
docker exec -it learn-tensorrt bash
```

Direct Docker attach is also supported because the container's main process is an interactive Bash
shell:

```bash
docker attach learn-tensorrt
```

Detach without stopping it by pressing `Ctrl-p`, then `Ctrl-q`. Typing `exit` or pressing `Ctrl-d`
in that attached main shell stops the container; use `docker start learn-tensorrt` to start it again.
For independent terminal sessions, prefer `docker exec -it learn-tensorrt bash`.

VS Code users can run `Dev Containers: Attach to Running Container...` and select
`learn-tensorrt`.

To recreate the container after changing the Dockerfile (the bind-mounted source is not deleted):

```bash
docker rm -f learn-tensorrt
# Re-run docker build and docker run above.
```

## Verify Inside The Container

Run:

```bash
cd /workspace/Learn-TensorRT
bash 00_environment_check/check_env.sh
```

Minimum pass conditions:

- the GPU and driver are visible through `nvidia-smi`;
- `nvcc` reports CUDA 13.0;
- `trtexec` and the TensorRT C++ and Python APIs report TensorRT 10.14.1.48;
- CMake, an ISO C++17 compiler, and OpenCV C++ development files are available;
- NVIDIA's PyTorch 25.11 and ModelOpt packages remain installed;
- Ultralytics, ONNX, and ONNX Runtime import successfully;
- the bind-mounted repository is writable by the non-root development user.

If a check fails, preserve the exact command output and classify the failure before making changes:
host driver, Docker daemon/runtime, upstream image, derived-image build, or project dependency.

## Agent Rules

- Do not install host components from inside the container.
- Do not change the pinned upstream image or replace its CUDA, TensorRT, PyTorch, or ModelOpt stack.
- Do not keep source or required artifacts only in the container's writable layer.
- Prefer Dockerfile and documentation changes over undocumented manual container mutation.
- Build engines, timing caches, golden outputs, and benchmarks only in this pinned environment, and
  record the GPU, driver, container image, CUDA, and TensorRT versions with the result.
