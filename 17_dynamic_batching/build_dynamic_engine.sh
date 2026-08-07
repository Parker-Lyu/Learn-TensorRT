#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ONNX_PATH=${1:-"${ROOT_DIR}/06_trtexec_engine/outputs/yolov8n_dynamic_autocast_fp16.onnx"}
OUTPUT_PATH=${2:-"${SCRIPT_DIR}/outputs/yolov8n_batch1_4_fp16.engine"}
TRTEXEC=${TRTEXEC:-/opt/tensorrt/bin/trtexec}
TIMING_CACHE=${TIMING_CACHE:-"${ROOT_DIR}/06_trtexec_engine/outputs/trtexec_timing.cache"}

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "Validated lesson 06 dynamic AutoCast ONNX model is required: ${ONNX_PATH}" >&2
  echo "Run lesson 05 export/validation, then python3 06_trtexec_engine/prepare_fp16_onnx.py." >&2
  exit 2
fi

mkdir -p "$(dirname -- "${OUTPUT_PATH}")" "$(dirname -- "${TIMING_CACHE}")"
"${TRTEXEC}" \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${OUTPUT_PATH}" \
  --stronglyTyped \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:2x3x640x640 \
  --maxShapes=images:4x3x640x640 \
  --timingCacheFile="${TIMING_CACHE}" \
  --builderOptimizationLevel=0 \
  --memPoolSize=workspace:2048
