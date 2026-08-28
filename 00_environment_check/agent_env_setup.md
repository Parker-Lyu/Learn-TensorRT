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
Ultralytics, ModelOpt ONNX, ONNX, ONNX Runtime, `onnx-graphsurgeon`, CUDA 13 CuPy,
`onnxslim`, and `onnxsim` versions. ModelOpt 0.37's published ONNX extra requests a CUDA 12 CuPy
distribution, so the Dockerfile installs the equivalent dependency set explicitly with
`cupy-cuda13x`; do not replace it with `cupy-cuda12x` in this CUDA 13 course image.

## Create The Persistent Container

Remove an old container before recreating it. The bind-mounted repository is not deleted:

```bash
docker rm -f learn-tensorrt 2>/dev/null || true

docker run -dit \
  --name learn-tensorrt \
  --gpus all \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount type=bind,source="$(pwd)",target=/workspace/Learn-TensorRT \
  --workdir /workspace/Learn-TensorRT \
  learn-tensorrt:25.11
```

Keep the default container unprivileged. Lesson 31 is the exception: when the host NVIDIA driver has
`RmProfilingAdminOnly=1`, recreate the persistent container with `--cap-add SYS_ADMIN` inserted after
`--gpus all`, run only the profiler process as container root, and restore bind-mounted output
ownership afterward. Do not change or reload the host driver merely to complete the lesson. Remove
the capability by recreating the normal container after profiling. A host configured to permit
non-admin profiling does not need this exception.

Check the policy and current container capability before recreating anything:

```bash
grep RmProfilingAdminOnly /proc/driver/nvidia/params
docker inspect learn-tensorrt --format '{{json .HostConfig.CapAdd}}'
```

When the policy is `1` and `CAP_SYS_ADMIN` is absent, use the complete profiling-container command
below from the repository root. The bind mount preserves repository files, but state stored only in
the old container's writable layer is discarded:

```bash
docker rm -f learn-tensorrt

docker run -dit \
  --name learn-tensorrt \
  --gpus all \
  --cap-add SYS_ADMIN \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --mount type=bind,source="$(pwd)",target=/workspace/Learn-TensorRT \
  --workdir /workspace/Learn-TensorRT \
  learn-tensorrt:25.11
```

Confirm the capability in Docker's configuration. The normal UID 1000 process intentionally has no
effective capabilities; invoke only the profiler workflow as container root:

```bash
docker inspect learn-tensorrt --format '{{json .HostConfig.CapAdd}}'

docker exec --user root learn-tensorrt bash -lc \
  'cd /workspace/Learn-TensorRT && \
   python3 31_nsight_compute_kernel_analysis/profile_kernels.py && \
   chown -R 1000:1000 31_nsight_compute_kernel_analysis/outputs'
```

Replace `1000:1000` with the UID/GID used to build the image when they differ. After profiling,
recreate the normal container with the first command in this section, which omits
`--cap-add SYS_ADMIN`. This avoids retaining a broad capability during ordinary lesson work.

`--network host` gives the container the host's network namespace. Lesson 24 runs the Triton
server with `--network host` on the host's ports, so the client inside the development container
can reach it at `localhost:8000` only when both share one network namespace. The trade-offs:

- Host networking disables port isolation; `-p` publish flags do not apply, and any port a course
  process binds is a host port. Avoid running two services on the same port.
- `--network host` works on Linux Docker hosts. On Docker Desktop for macOS or Windows it has
  limited, version-dependent support; this course targets a native Linux host.
- Existing containers keep their original network mode. Recreate the container as shown above to
  apply the change; the bind-mounted repository is not affected, but state stored only in the
  container's writable layer is lost.

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
- ModelOpt 0.37 AutoCast requires its ONNX optional dependencies and is aligned here with ONNX
  1.19.1 and ONNX Runtime 1.22.0. ModelOpt's Linux `onnx` extra names `onnxruntime-gpu`, but the
  course intentionally installs the CPU `onnxruntime` distribution. Lesson 05 and AutoCast use
  the CPU provider for reproducible ONNX validation; TensorRT is the GPU deployment runtime. This
  is a course architecture decision, not a claim that the GPU distribution is unusable. An
  ONNX Runtime CUDA-EP experiment must use a separately validated dependency profile, and must not
  replace the course baseline or install both distributions together. Keeping the CPU distribution
  also satisfies tools that check Ultralytics' export dependency by distribution name.
  Installing GraphSurgeon alone beside ONNX 1.21 is not a valid repair; rebuild the pinned image
  instead.
- The upstream image's `nvidia-resiliency-ext` metadata requires the deprecated `pynvml`
  distribution even though the image already supplies the maintained `nvidia-ml-py` bindings.
  Consequently `pip check` reports that one metadata warning. Do not install `pynvml` merely to
  silence it: doing so makes current PyTorch emit a deprecation warning on every import.
- `nvidia-smi` reports the maximum CUDA version supported by the driver. It may be newer than the
  container's CUDA 13.0 `nvcc`; this is expected.

## Agent Rules

- Prefer reproducible Dockerfile, check-script, and documentation fixes over manual container
  mutation.
- Do not keep source or required resources only in the container writable layer.
- Run GPU-, CUDA-, and TensorRT-dependent verification in this container.
- If a required GPU, image, or network resource is unavailable, run the strongest available checks
  and state exactly what was not verified.
