#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
DEVELOPMENT_IMAGE=${DEVELOPMENT_IMAGE:-learn-tensorrt:25.11}
RUNTIME_BASE_IMAGE=${RUNTIME_BASE_IMAGE:-nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04}
RUNTIME_IMAGE=${RUNTIME_IMAGE:-learn-tensorrt-runtime:10.14}
ENGINE="${SCRIPT_DIR}/outputs/yolov8n_static_autocast_fp16.engine"

command -v docker >/dev/null || { echo "Docker is required" >&2; exit 2; }
if [[ ! -f "${ENGINE}" ]]; then
  echo "Missing ${ENGINE}" >&2
  echo "Run 32_final_portfolio_case_study/build_delivery_engine.sh first." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
docker build \
  --build-arg "DEVELOPMENT_IMAGE=${DEVELOPMENT_IMAGE}" \
  --build-arg "RUNTIME_IMAGE=${RUNTIME_BASE_IMAGE}" \
  -f "${SCRIPT_DIR}/Dockerfile" \
  -t "${RUNTIME_IMAGE}" \
  "${ROOT_DIR}"

docker image inspect "${DEVELOPMENT_IMAGE}" "${RUNTIME_BASE_IMAGE}" "${RUNTIME_IMAGE}" \
  --format '{{json .RepoTags}} {{.Id}} {{.Size}}' | tee "${OUTPUT_DIR}/docker_image_sizes.txt"

python3 - "${OUTPUT_DIR}/platform_manifest.json" "${DEVELOPMENT_IMAGE}" \
  "${RUNTIME_BASE_IMAGE}" "${RUNTIME_IMAGE}" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(args):
    try:
        result = subprocess.run(args, text=True, capture_output=True, check=False)
    except OSError as error:
        return {"command": args, "returncode": None, "stdout": "", "stderr": str(error)}
    return {"command": args, "returncode": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


output, development, runtime_base, runtime = sys.argv[1:]
manifest = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "images": {
        name: command(["docker", "image", "inspect", image, "--format", "{{json .}}"])
        for name, image in (("development", development), ("runtime_base", runtime_base),
                            ("runtime", runtime))
    },
    "gpu": command(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
                    "--format=csv,noheader,nounits"]),
}
Path(output).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY
