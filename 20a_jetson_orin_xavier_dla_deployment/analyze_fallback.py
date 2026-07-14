#!/usr/bin/env python3
"""Extract an auditable GPU/DLA assignment summary from a detailed TensorRT build log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def analyze(text: str) -> dict:
    dla = len(re.findall(r"DeviceType[^\n]*DLA|device[^\n]*DLA", text, re.IGNORECASE))
    gpu = len(re.findall(r"DeviceType[^\n]*GPU|device[^\n]*GPU", text, re.IGNORECASE))
    warnings = [line for line in text.splitlines()
                if "fallback" in line.lower() or "cannot run on DLA" in line]
    return {"dla_assignment_mentions": dla, "gpu_assignment_mentions": gpu,
            "fallback_warnings": warnings}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=ROOT / "outputs/dla_build.log")
    args = parser.parse_args()
    if not args.log.is_file():
        raise FileNotFoundError("build the DLA engine on Jetson first")
    print(json.dumps(analyze(args.log.read_text(encoding="utf-8", errors="replace")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
