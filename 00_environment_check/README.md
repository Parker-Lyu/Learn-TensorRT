# 00 - Environment Check

Goal: confirm that the GPU, CUDA, TensorRT, C++, and Python environment required by later lessons
is working.

The course uses a development container derived from:

```text
nvcr.io/nvidia/pytorch:25.11-py3
```

Build and container-creation commands are documented in
[`agent_env_setup.md`](agent_env_setup.md). If a course maintainer or coding agent has already
created the environment, do not reinstall its dependencies.

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
