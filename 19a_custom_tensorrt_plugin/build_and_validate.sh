#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${SCRIPT_DIR}/build" -j
python3 "${SCRIPT_DIR}/create_plugin_model.py"
trtexec \
  --staticPlugins="${SCRIPT_DIR}/build/libscale_shift_plugin.so" \
  --onnx="${SCRIPT_DIR}/outputs/scale_shift.onnx" \
  --saveEngine="${SCRIPT_DIR}/outputs/scale_shift.engine" \
  --skipInference
"${SCRIPT_DIR}/build/validate_plugin" "${SCRIPT_DIR}/outputs/scale_shift.engine"
