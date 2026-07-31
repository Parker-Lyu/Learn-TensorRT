#!/usr/bin/env python3
"""Run Lesson 21 CPU tests under TSAN with the documented kernel workaround."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="learn-tensorrt:25.11")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "22_pipeline_performance_report/outputs/tsan.json")
    args = parser.parse_args()
    command = [
        "docker", "run", "--rm", "--security-opt", "seccomp=unconfined",
        "-v", f"{ROOT}:/workspace/Learn-TensorRT", "-w", "/workspace/Learn-TensorRT",
        args.image, "bash", "-lc",
        "set -e; "
        "setarch x86_64 -R 21_integrated_tensorrt_video_pipeline/build-tsan/"
        "integrated_pipeline_core_tests; "
        "setarch x86_64 -R 21_integrated_tensorrt_video_pipeline/build-tsan/"
        "integrated_frame_scheduler_tests",
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    evidence = {"command": command, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr,
                "tool_started": "unexpected memory mapping" not in result.stderr}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
