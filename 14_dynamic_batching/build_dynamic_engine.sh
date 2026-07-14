#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ONNX_PATH=${1:-"${ROOT_DIR}/05_torch_to_onnx/outputs/yolov8n_dynamic.onnx"}
OUTPUT_PATH=${2:-"${SCRIPT_DIR}/outputs/yolov8n_batch1_4_fp16.engine"}
TRTEXEC=${TRTEXEC:-/opt/tensorrt/bin/trtexec}
TIMING_CACHE=${TIMING_CACHE:-"${ROOT_DIR}/06_trtexec_engine/outputs/trtexec_timing.cache"}

mkdir -p "$(dirname -- "${OUTPUT_PATH}")"
"${TRTEXEC}" \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${OUTPUT_PATH}" \
  --fp16 \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:2x3x640x640 \
  --maxShapes=images:4x3x640x640 \
  --timingCacheFile="${TIMING_CACHE}" \
  --minTiming=1 \
  --avgTiming=1 \
  --builderOptimizationLevel=0 \
  --memPoolSize=workspace:2048
