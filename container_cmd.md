# TensorRT Container Commands

This project uses the NVIDIA TensorRT container image:

```bash
nvcr.io/nvidia/tensorrt:23.10-py3
```

## Start the container

```bash
docker run -d --name learn-tensorrt --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /home/parker/Projects/Learn-TensorRT:/workspace/Learn-TensorRT -w /workspace/Learn-TensorRT nvcr.io/nvidia/tensorrt:23.10-py3 sleep infinity
```

Explanation:

- `docker run`: start a new container from an image.
- `-d`: run the container in the background.
- `--name learn-tensorrt`: name the container `learn-tensorrt`, so it can be referenced easily later.
- `--gpus all`: expose all NVIDIA GPUs from the host to the container.
- `--ipc=host`: use the host IPC namespace, which is commonly used for deep learning workloads to avoid shared-memory limits.
- `--ulimit memlock=-1`: allow unlimited locked memory.
- `--ulimit stack=67108864`: set the stack size to 64 MB.
- `-v /home/parker/Projects/Learn-TensorRT:/workspace/Learn-TensorRT`: mount the current project directory into the container.
- `-w /workspace/Learn-TensorRT`: set the container working directory to the mounted project directory.
- `nvcr.io/nvidia/tensorrt:23.10-py3`: use the NVIDIA TensorRT 23.10 Python 3 image.
- `sleep infinity`: keep the container running so it can be entered later with `docker exec`.

## Enter the container

```bash
docker exec -it learn-tensorrt bash
```

Explanation:

- `docker exec`: run a command inside an existing container.
- `-it`: allocate an interactive terminal.
- `learn-tensorrt`: the target container name.
- `bash`: start a Bash shell inside the container.

## Stop the container

```bash
docker stop learn-tensorrt
```

Explanation:

- `docker stop`: stop a running container gracefully.
- `learn-tensorrt`: the container name to stop.
