#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${SCRIPT_DIR}/build" -j
python3 "${SCRIPT_DIR}/../25_onnx_graph_surgery_plugin/create_demo_model.py"
/opt/tensorrt/bin/trtexec \
  --stronglyTyped \
  --staticPlugins="${SCRIPT_DIR}/build/libacme_swish_plugin.so" \
  --onnx="${SCRIPT_DIR}/../25_onnx_graph_surgery_plugin/outputs/unsupported_swish.onnx" \
  --saveEngine="${SCRIPT_DIR}/outputs/acme_swish.engine" \
  --skipInference
"${SCRIPT_DIR}/build/validate_plugin" "${SCRIPT_DIR}/outputs/acme_swish.engine"
