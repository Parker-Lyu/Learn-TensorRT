#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ONNX_PATH=${1:-"${ROOT_DIR}/05_torch_to_onnx/outputs/yolov8n_dynamic.onnx"}
OUTPUT_PATH=${2:-"${SCRIPT_DIR}/outputs/yolov8n_batch1_4_fp16.engine"}
REFERENCE_INPUT=${REFERENCE_INPUT:-"${ROOT_DIR}/05_torch_to_onnx/outputs/input_nchw_float32.npy"}
AUTOCAST_ONNX=${AUTOCAST_ONNX:-"${SCRIPT_DIR}/outputs/yolov8n_dynamic_autocast_fp16.onnx"}
AUTOCAST_INPUT=${AUTOCAST_INPUT:-"${SCRIPT_DIR}/outputs/autocast_input.npz"}
DEPS_DIR=${DEPS_DIR:-"${SCRIPT_DIR}/.deps"}
TRTEXEC=${TRTEXEC:-/opt/tensorrt/bin/trtexec}
TIMING_CACHE=${TIMING_CACHE:-"${ROOT_DIR}/06_trtexec_engine/outputs/trtexec_timing.cache"}

if [[ ! -f "${ONNX_PATH}" || ! -f "${REFERENCE_INPUT}" ]]; then
  echo "Lesson 05 dynamic ONNX and validated input tensor are required." >&2
  echo "Run the lesson 05 export and validation commands first." >&2
  exit 2
fi
if [[ ! -d "${DEPS_DIR}/onnx_graphsurgeon" ]]; then
  echo "AutoCast dependencies are missing. Run ${SCRIPT_DIR}/setup_autocast_deps.sh" >&2
  exit 2
fi

mkdir -p "$(dirname -- "${OUTPUT_PATH}")" "$(dirname -- "${TIMING_CACHE}")"
PYTHONPATH="${DEPS_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  python3 "${SCRIPT_DIR}/prepare_autocast_input.py" \
    --input "${REFERENCE_INPUT}" \
    --output "${AUTOCAST_INPUT}" \
    --batch 2
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
  --minShapes=images:1x3x640x640 \
  --optShapes=images:2x3x640x640 \
  --maxShapes=images:4x3x640x640 \
  --timingCacheFile="${TIMING_CACHE}" \
  --builderOptimizationLevel=0 \
  --memPoolSize=workspace:2048
