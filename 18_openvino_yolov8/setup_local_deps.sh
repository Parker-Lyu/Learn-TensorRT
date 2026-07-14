#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python3 -m pip install \
  --target "${SCRIPT_DIR}/.deps" \
  --requirement "${SCRIPT_DIR}/requirements.txt"
