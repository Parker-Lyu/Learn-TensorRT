#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rm -rf "${SCRIPT_DIR}/.deps"
python3 -m pip install --target "${SCRIPT_DIR}/.deps" -r "${SCRIPT_DIR}/requirements.txt"
