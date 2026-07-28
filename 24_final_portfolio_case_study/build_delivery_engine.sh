#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ONNX_PATH=${ONNX_PATH:-"${ROOT_DIR}/05_torch_to_onnx/outputs/yolov8n.onnx"}
REFERENCE_INPUT=${REFERENCE_INPUT:-"${ROOT_DIR}/05_torch_to_onnx/outputs/input_nchw_float32.npy"}
AUTOCAST_ONNX=${AUTOCAST_ONNX:-"${SCRIPT_DIR}/outputs/yolov8n_static_autocast_fp16.onnx"}
AUTOCAST_INPUT=${AUTOCAST_INPUT:-"${SCRIPT_DIR}/outputs/autocast_input.npz"}
OUTPUT_PATH=${OUTPUT_PATH:-"${SCRIPT_DIR}/outputs/yolov8n_static_autocast_fp16.engine"}
DEPS_DIR=${DEPS_DIR:-"${ROOT_DIR}/14_dynamic_batching/.deps"}
TRTEXEC=${TRTEXEC:-/opt/tensorrt/bin/trtexec}
TIMING_CACHE=${TIMING_CACHE:-"${ROOT_DIR}/06_trtexec_engine/outputs/trtexec_timing.cache"}

if [[ ! -f "${ONNX_PATH}" || ! -f "${REFERENCE_INPUT}" ]]; then
  echo "Lesson 05 static ONNX and validated input tensor are required." >&2
  exit 2
fi
if [[ ! -d "${DEPS_DIR}/onnx_graphsurgeon" ]]; then
  echo "AutoCast dependencies are missing. Run 14_dynamic_batching/setup_autocast_deps.sh" >&2
  exit 2
fi

mkdir -p "${SCRIPT_DIR}/outputs" "$(dirname -- "${TIMING_CACHE}")"
PYTHONPATH="${DEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 "${ROOT_DIR}/14_dynamic_batching/prepare_autocast_input.py" \
    --input "${REFERENCE_INPUT}" \
    --output "${AUTOCAST_INPUT}" \
    --batch 1
PYTHONPATH="${DEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 -m modelopt.onnx.autocast \
    --onnx_path "${ONNX_PATH}" \
    --output_path "${AUTOCAST_ONNX}" \
    --low_precision_type fp16 \
    --keep_io_types \
    --calibration_data "${AUTOCAST_INPUT}" \
    --providers cpu \
    --nodes_to_exclude '/model\.22/.*'

"${TRTEXEC}" \
  --onnx="${AUTOCAST_ONNX}" \
  --saveEngine="${OUTPUT_PATH}" \
  --stronglyTyped \
  --timingCacheFile="${TIMING_CACHE}" \
  --builderOptimizationLevel=0 \
  --memPoolSize=workspace:2048 \
  --skipInference
