#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
command -v docker >/dev/null || { echo "Docker is required" >&2; exit 2; }
docker build -f "${SCRIPT_DIR}/Dockerfile" -t learn-tensorrt-runtime "${ROOT_DIR}"
docker image inspect nvcr.io/nvidia/tensorrt:23.10-py3 learn-tensorrt-runtime \
  --format '{{.RepoTags}} {{.Size}}' | tee "${SCRIPT_DIR}/outputs/docker_image_sizes.txt"
