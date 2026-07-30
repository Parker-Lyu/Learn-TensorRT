#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
mkdir -p "${SCRIPT_DIR}/models"
"${TRTEXEC:-trtexec}" \
  --onnx="${ROOT_DIR}/05_torch_to_onnx/outputs/yolov8n_dynamic.onnx" \
  --saveEngine="${SCRIPT_DIR}/models/yolov8n_b2_fp16.engine" \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:2x3x640x640 \
  --maxShapes=images:4x3x640x640
cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${SCRIPT_DIR}/build" -j
