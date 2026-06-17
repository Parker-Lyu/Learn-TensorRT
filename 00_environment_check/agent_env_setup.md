# Agent Guide - TensorRT Dev Container Environment

This document is written for an AI coding agent preparing or verifying the development environment for this repository.

The user develops with VS Code Dev Containers attached to a running Docker container. The container is the development environment. Do not assume CUDA, cuDNN, TensorRT, OpenCV, or compiler tools should be installed on the host.

## Non-Negotiable Image

Use this image:

```bash
nvcr.io/nvidia/tensorrt:23.10-py3
```

Do not replace it with another TensorRT, CUDA, Ubuntu, or PyTorch image. This course is pinned to this image because it provides a stable TensorRT C++ learning environment:

- Ubuntu 22.04 base
- CUDA 12.2 era toolchain
- cuDNN 8.9 era runtime
- TensorRT 8.6.1 era runtime and development files

## Intended Workflow

1. The host machine runs Ubuntu.
2. The host machine has the NVIDIA driver installed.
3. The host machine has Docker Engine installed from Docker's official apt repository.
4. The host machine has NVIDIA Container Toolkit installed and configured for Docker.
5. The repository is bind-mounted into the container.
6. VS Code connects to the container through Dev Containers.
7. All build, test, CMake, CUDA, TensorRT, and Python commands are run inside the container.

## Host Responsibilities

The host should provide only the minimum base needed for GPU containers:

- NVIDIA driver
- Docker Engine
- Docker Buildx and Compose plugins
- NVIDIA Container Toolkit
- A persistent repository directory mounted into the container

Avoid installing a full CUDA Toolkit or TensorRT stack directly on the host unless the user explicitly asks for host-native development.

## Host Setup Reference

These commands are for the user or a privileged automation step on the host. Do not run them from inside the container.

Verify the host driver first:

```bash
nvidia-smi
```

Install Docker Engine from Docker's official repository, not `apt install docker.io`:

```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do
  sudo apt-get remove "$pkg"
done

sudo apt-get update
sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker "$USER"
```

After changing Docker group membership, the user should log out and back in, or run:

```bash
newgrp docker
```

Install NVIDIA Container Toolkit:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Pull the TensorRT image:

```bash
docker pull nvcr.io/nvidia/tensorrt:23.10-py3
```

## Container Launch Pattern

For a persistent container that VS Code Dev Containers can attach to:

```bash
docker run -d \
  --name learn-tensorrt \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /workspace/Projects/Learn-TensorRT:/workspace/Projects/Learn-TensorRT \
  -w /workspace/Projects/Learn-TensorRT \
  nvcr.io/nvidia/tensorrt:23.10-py3 \
  sleep infinity
```

If the repository is stored elsewhere on the host, keep the same container image and adjust only the bind mount path.

Then attach VS Code to the running container with Dev Containers:

```text
Dev Containers: Attach to Running Container...
```

Choose `learn-tensorrt`.

## In-Container Verification

Inside the Dev Container, install the Python dependencies required by the early lessons:

```bash
python3 -m pip install --no-cache-dir \
  ultralytics \
  onnx==1.21.0 \
  onnxruntime==1.23.2
```

If the network is slow or unstable, especially in mainland China, use a domestic PyPI mirror such as Aliyun:

```bash
python3 -m pip install \
  --no-cache-dir \
  --default-timeout 120 \
  --retries 10 \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com \
  ultralytics \
  onnx==1.21.0 \
  onnxruntime==1.23.2
```

Install optional ONNX graph simplifiers here too if later export experiments need graph
simplification:

```bash
python3 -m pip install --no-cache-dir onnxslim onnxsim
```

Then run the environment check:

```bash
bash 00_environment_check/check_env.sh
```

Minimum pass conditions:

- `nvidia-smi` prints the RTX 2060 or another NVIDIA GPU.
- `nvcc --version` prints a CUDA compiler version.
- `trtexec --help` runs and reports TensorRT.
- `libnvinfer.so*` exists under a standard library path.
- `cmake`, `g++`, and `python3` are available.
- Python can import `ultralytics`, `onnx`, and `onnxruntime`.
- The repository path is writable.

## Agent Rules

- Do not install Docker, drivers, or NVIDIA Container Toolkit from inside the container.
- Do not change the base image unless the user explicitly asks.
- Do not store source code only inside an ephemeral container filesystem.
- Prefer adding reproducible scripts and documentation to this repository over changing the host manually.
- If a check fails, report the failing command and likely layer: host driver, Docker GPU runtime, container image, or project dependency.

## Python Dependencies

The base TensorRT image already contains many packages, but this course requires ONNX export and validation tools as part of the environment baseline.

Install them inside the Dev Container:

```bash
python3 -m pip install --no-cache-dir \
  ultralytics \
  onnx==1.21.0 \
  onnxruntime==1.23.2
```

If the network is slow or unstable, especially in mainland China, use a domestic mirror:

```bash
python3 -m pip install \
  --no-cache-dir \
  --default-timeout 120 \
  --retries 10 \
  -i https://mirrors.aliyun.com/pypi/simple/ \
  --trusted-host mirrors.aliyun.com \
  ultralytics \
  onnx==1.21.0 \
  onnxruntime==1.23.2
```

Optional graph simplification packages belong in this baseline setup when needed, not in individual
lesson READMEs:

```bash
python3 -m pip install --no-cache-dir onnxslim onnxsim
```
