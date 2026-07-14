#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ONNX_PATH=${1:-"${ROOT_DIR}/05_torch_to_onnx/outputs/yolov8n.onnx"}
OUTPUT_DIR="${SCRIPT_DIR}/outputs"

python3 "${SCRIPT_DIR}/check_platform.py" --require-jetson
mkdir -p "${OUTPUT_DIR}"

trtexec \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${OUTPUT_DIR}/yolov8n_jetson_gpu_fp16.engine" \
  --fp16 \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  > "${OUTPUT_DIR}/gpu_build.log" 2>&1

trtexec \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${OUTPUT_DIR}/yolov8n_jetson_dla_fp16.engine" \
  --fp16 \
  --useDLACore=0 \
  --allowGPUFallback \
  --profilingVerbosity=detailed \
  --dumpLayerInfo \
  > "${OUTPUT_DIR}/dla_build.log" 2>&1

echo "Built target-local GPU and DLA-with-fallback engines under ${OUTPUT_DIR}"
