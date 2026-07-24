# Agent Guide - Development Environment

This document is for coding agents configuring or troubleshooting the course container. It contains
only the environment constraints and commands needed for that work. The learner-facing entry point
is this lesson's `README.md`.

## Fixed Environment

- Upstream image: `nvcr.io/nvidia/pytorch:25.11-py3`
- CUDA Toolkit: 13.0
- TensorRT: 10.14.1.48
- Dockerfile: `docker/Dockerfile.dev`
- Derived image: `learn-tensorrt:25.11`
- Persistent container: `learn-tensorrt`
- Repository path in the container: `/workspace/Learn-TensorRT`

Do not independently upgrade or replace CUDA, TensorRT, PyTorch, or ModelOpt from the upstream
image.

## Check The Host First

The host provides only the NVIDIA driver, Docker Engine, and NVIDIA Container Toolkit. Do not
install the course CUDA, TensorRT, or OpenCV stack on the host.

```bash
nvidia-smi
docker version
docker info --format '{{json .Runtimes}}' | grep nvidia
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 nvidia-smi
```

If these pass, do not reinstall host components. Classify failures before changing anything:

- Host `nvidia-smi` failure: NVIDIA driver layer.
- Missing Docker `nvidia` runtime: NVIDIA Container Toolkit layer.
- GPU container test failure only: Docker GPU runtime or driver compatibility layer.

The remediation commands below target supported Ubuntu hosts. Run only the section for the missing
layer. Privileged commands require user authorization, and a driver change may require a reboot.

### Install Or Repair The NVIDIA Driver

Skip this section when host `nvidia-smi` already works and reports driver 580.95.05 or newer. Do not
install a host CUDA Toolkit: the course toolkit comes from the container image.

Use Ubuntu's driver selection instead of hard-coding a driver branch that will become stale:

```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common
ubuntu-drivers devices
sudo ubuntu-drivers install
sudo reboot
```

After the host restarts:

```bash
nvidia-smi
```

Do not mix Ubuntu apt-managed drivers with NVIDIA `.run` installers. If another installation method
is already present, inspect it and report the conflict before changing the driver.

### Install Docker Engine

Skip this section when `docker version` already succeeds. Use Docker's official Ubuntu repository,
not the Ubuntu `docker.io` package:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

sudo apt-get update
sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

sudo usermod -aG docker "$USER"
```

Log out and back in to apply Docker group membership, or start a temporary shell with:

```bash
newgrp docker
```

Then verify:

```bash
docker version
docker run --rm hello-world
```

If apt reports a conflict with an existing Docker or containerd package, inspect the installed
packages and active containers before removing anything. Do not automatically delete another
working container stack.

### Install NVIDIA Container Toolkit

Run this section only when the host driver and Docker work but Docker has no usable NVIDIA runtime:

```bash
sudo apt-get update
sudo apt-get install -y curl gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor --yes \
  -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -sL \
  https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed \
    's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify the complete host-to-container GPU path before building the course image:

```bash
docker info --format '{{json .Runtimes}}' | grep nvidia
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 nvidia-smi
```

## Build The Image

From the repository root:

```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3
docker build \
  --build-arg USER_UID="$(id -u)" \
  --build-arg USER_GID="$(id -g)" \
  -f docker/Dockerfile.dev \
  -t learn-tensorrt:25.11 .
```

The UID/GID arguments prevent root-owned files in the bind mount. The Dockerfile pins the required
Ultralytics, ONNX, ONNX Runtime, `onnxslim`, and `onnxsim` versions.

## Create The Persistent Container

Remove an old container before recreating it. The bind-mounted repository is not deleted:

```bash
docker rm -f learn-tensorrt 2>/dev/null || true

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

Start or enter it with:

```bash
docker start learn-tensorrt                 # If it is stopped
docker exec -it learn-tensorrt bash         # Preferred: independent shell
docker attach learn-tensorrt                # Attach to the main shell
```

For `docker attach`, use `Ctrl-p`, then `Ctrl-q` to detach. Running `exit` in the main shell stops
the container.

## Verify

```bash
docker exec -it learn-tensorrt bash
cd /workspace/Learn-TensorRT
bash 00_environment_check/check_env.sh
```

The check must end with:

```text
[PASS] Environment checks passed.
```

Preserve complete failure output. Do not work around failures by arbitrarily upgrading packages.

## Known Compatibility Constraints

- The upstream image already has a UID 1000 user. The Dockerfile reuses an existing target UID
  instead of creating a duplicate.
- Ubuntu's full OpenCV development package introduces another UCX runtime through VTK/OpenMPI. The
  Dockerfile's `LD_PRELOAD` keeps PyTorch on the upstream HPC-X UCX libraries. Removing it causes
  undefined-symbol errors during `import torch`.
- CUDA 13 changed the `cudaMemPrefetchAsync` API. Lesson 04 contains the CUDA 13 compatibility code;
  do not downgrade CUDA to make old call signatures compile.
- `nvidia-smi` reports the maximum CUDA version supported by the driver. It may be newer than the
  container's CUDA 13.0 `nvcc`; this is expected.

## Agent Rules

- Prefer reproducible Dockerfile, check-script, and documentation fixes over manual container
  mutation.
- Do not keep source or required resources only in the container writable layer.
- Run GPU-, CUDA-, and TensorRT-dependent verification in this container.
- If a required GPU, image, or network resource is unavailable, run the strongest available checks
  and state exactly what was not verified.
