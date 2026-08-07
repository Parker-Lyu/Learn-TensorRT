#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
ONNX_PATH=${ONNX_PATH:-"${ROOT_DIR}/06_trtexec_engine/outputs/yolov8n_static_autocast_fp16.onnx"}
OUTPUT_PATH=${OUTPUT_PATH:-"${SCRIPT_DIR}/outputs/yolov8n_static_fp16_strong.engine"}
TRTEXEC=${TRTEXEC:-/opt/tensorrt/bin/trtexec}
TIMING_CACHE=${TIMING_CACHE:-"${ROOT_DIR}/06_trtexec_engine/outputs/trtexec_timing.cache"}
VALIDATION_REPORT=${VALIDATION_REPORT:-"${ROOT_DIR}/06_trtexec_engine/outputs/static_fp16_onnx_validation.json"}

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "Validated lesson 06 static AutoCast ONNX model is required: ${ONNX_PATH}" >&2
  exit 2
fi
if [[ ! -f "${VALIDATION_REPORT}" ]]; then
  echo "Lesson 06 static FP16 validation report is required: ${VALIDATION_REPORT}" >&2
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
    raise SystemExit("lesson 06 static FP16 validation did not pass")
if hashlib.sha256(model.read_bytes()).hexdigest() != data.get("converted_sha256"):
    raise SystemExit("static FP16 ONNX does not match the validated lesson 06 artifact")
PY

mkdir -p "${SCRIPT_DIR}/outputs" "$(dirname -- "${TIMING_CACHE}")"
"${TRTEXEC}" \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${OUTPUT_PATH}" \
  --stronglyTyped \
  --timingCacheFile="${TIMING_CACHE}" \
  --builderOptimizationLevel=0 \
  --memPoolSize=workspace:2048 \
  --skipInference
