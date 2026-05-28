# 00 - Environment Check

Goal: record the machine, driver, CUDA, TensorRT, compiler, Python, and OpenCV environment before running deployment experiments.

This repository is developed inside a TensorRT Docker container, and VS Code connects to that container through Dev Containers. The host machine should keep the setup minimal: NVIDIA driver, Docker Engine, and NVIDIA Container Toolkit. CUDA, cuDNN, TensorRT, compiler tools, and Python packages should live inside the container.

Required container image:

```bash
nvcr.io/nvidia/tensorrt:23.10-py3
```

Do not change this image for this course unless a later lesson explicitly asks for a separate experiment.

For AI coding agents that need to prepare or verify the development environment, read:

```bash
00_environment_check/agent_env_setup.md
```

Deliverables:

- `check_env.sh`
- `env_report.md`

Run the environment check from inside the Dev Container:

```bash
cd /workspace/Projects/Learn-TensorRT
bash 00_environment_check/check_env.sh
```

Key manual commands:

```bash
nvidia-smi
nvcc --version
trtexec --help
cmake --version
g++ --version
python3 --version
```

Acceptance criteria:

- VS Code is attached to the TensorRT container through Dev Containers.
- The project directory is mounted into the container and is writable.
- `nvidia-smi` works inside the container.
- `nvcc --version` works inside the container.
- TensorRT C++ libraries such as `libnvinfer.so*` are visible.
- Python can import `onnx` and `onnxruntime`.
- Basic C++ build tools are available.
