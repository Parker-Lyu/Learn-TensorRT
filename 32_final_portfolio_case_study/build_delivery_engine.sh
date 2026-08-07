#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ONNX_PATH=${ONNX_PATH:-"${ROOT_DIR}/06_trtexec_engine/outputs/yolov8n_static_autocast_fp16.onnx"}
OUTPUT_PATH=${OUTPUT_PATH:-"${SCRIPT_DIR}/outputs/yolov8n_static_fp16_strong.engine"}
TRTEXEC=${TRTEXEC:-/opt/tensorrt/bin/trtexec}
TIMING_CACHE=${TIMING_CACHE:-"${ROOT_DIR}/06_trtexec_engine/outputs/trtexec_timing.cache"}

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "Validated lesson 06 static AutoCast ONNX model is required: ${ONNX_PATH}" >&2
  exit 2
fi

mkdir -p "${SCRIPT_DIR}/outputs" "$(dirname -- "${TIMING_CACHE}")"
"${TRTEXEC}" \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${OUTPUT_PATH}" \
  --stronglyTyped \
  --timingCacheFile="${TIMING_CACHE}" \
  --builderOptimizationLevel=0 \
  --memPoolSize=workspace:2048 \
  --skipInference
