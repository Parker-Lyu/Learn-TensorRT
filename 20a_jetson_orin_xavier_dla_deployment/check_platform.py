#!/usr/bin/env python3
"""Record Jetson/JetPack/TensorRT/DLA compatibility before building target engines."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def command(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return (result.stdout + result.stderr).strip()


def detect() -> dict:
    release_file = Path("/etc/nv_tegra_release")
    l4t_release = release_file.read_text(encoding="utf-8").strip() if release_file.exists() else ""
    try:
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.ERROR)
        builder = trt.Builder(logger)
        tensorrt_version = trt.__version__
        dla_cores = int(builder.num_DLA_cores)
    except Exception as error:
        tensorrt_version = f"unavailable: {error}"
        dla_cores = 0
    machine = platform.machine()
    is_jetson = machine == "aarch64" and (release_file.exists() or Path("/etc/nv_boot_control.conf").exists())
    return {
        "machine": machine,
        "is_jetson": is_jetson,
        "l4t_release": l4t_release,
        "jetpack_package": command(["dpkg-query", "-W", "-f=${Version}", "nvidia-jetpack"]),
        "cuda_compiler": command(["nvcc", "--version"]),
        "tensorrt": tensorrt_version,
        "dla_cores": dla_cores,
        "kernel": platform.release(),
        "engine_portability_warning":
            "TensorRT engines must be rebuilt on the target JetPack/TensorRT/device combination",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/platform_manifest.json")
    parser.add_argument("--require-jetson", action="store_true")
    args = parser.parse_args()
    report = detect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.require_jetson and (not report["is_jetson"] or report["dla_cores"] < 1):
        print("Jetson with at least one DLA core is required for this command")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
