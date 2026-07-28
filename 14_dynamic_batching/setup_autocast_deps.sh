#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEPS_DIR="${SCRIPT_DIR}/.deps"

python3 - "${DEPS_DIR}" <<'PY'
import shutil
import sys

shutil.rmtree(sys.argv[1], ignore_errors=True)
PY
python3 -m pip install \
  --requirement "${SCRIPT_DIR}/requirements.txt" \
  --target "${DEPS_DIR}"
