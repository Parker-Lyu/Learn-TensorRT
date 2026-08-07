#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ONNX_PATH=${1:-"${ROOT_DIR}/06_trtexec_engine/outputs/yolov8n_dynamic_autocast_fp16.onnx"}
OUTPUT_PATH=${2:-"${SCRIPT_DIR}/outputs/yolov8n_batch1_4_fp16.engine"}
TRTEXEC=${TRTEXEC:-/opt/tensorrt/bin/trtexec}
TIMING_CACHE=${TIMING_CACHE:-"${ROOT_DIR}/06_trtexec_engine/outputs/trtexec_timing.cache"}
VALIDATION_REPORT=${VALIDATION_REPORT:-"${ROOT_DIR}/06_trtexec_engine/outputs/dynamic_fp16_onnx_validation.json"}

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "Validated lesson 06 dynamic AutoCast ONNX model is required: ${ONNX_PATH}" >&2
  echo "Run lesson 05 export/validation, then python3 06_trtexec_engine/prepare_fp16_onnx.py." >&2
  exit 2
fi
if [[ ! -f "${VALIDATION_REPORT}" ]]; then
  echo "Lesson 06 dynamic FP16 validation report is required: ${VALIDATION_REPORT}" >&2
  exit 2
fi
python3 - "${ONNX_PATH}" "${VALIDATION_REPORT}" <<'PY'
import hashlib
import json
import pathlib
import sys

model, report = map(pathlib.Path, sys.argv[1:])
data = json.loads(report.read_text())
if not data.get("comparison", {}).get("passed", False):
    raise SystemExit("lesson 06 dynamic FP16 validation did not pass")
if hashlib.sha256(model.read_bytes()).hexdigest() != data.get("converted_sha256"):
    raise SystemExit("dynamic FP16 ONNX does not match the validated lesson 06 artifact")
PY

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
