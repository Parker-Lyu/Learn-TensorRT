#!/usr/bin/env python3
"""Build TensorRT engines with trtexec and keep reproducible benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATIC_ONNX = REPO_ROOT / "05_torch_to_onnx" / "outputs" / "yolov8n.onnx"
DEFAULT_DYNAMIC_ONNX = REPO_ROOT / "05_torch_to_onnx" / "outputs" / "yolov8n_dynamic.onnx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "06_trtexec_engine" / "outputs"


@dataclass(frozen=True)
class EngineBuild:
    name: str
    onnx_path: Path
    engine_path: Path
    log_path: Path
    times_path: Path
    layer_info_path: Path
    profile_path: Path
    fp16: bool
    dynamic: bool
    shapes: str | None = None
    min_shapes: str | None = None
    opt_shapes: str | None = None
    max_shapes: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FP32/FP16 TensorRT engines from lesson 05 ONNX artifacts."
    )
    parser.add_argument("--onnx", type=Path, default=DEFAULT_STATIC_ONNX, help="Static ONNX path.")
    parser.add_argument(
        "--dynamic-onnx",
        type=Path,
        default=DEFAULT_DYNAMIC_ONNX,
        help="Dynamic ONNX path. If missing, dynamic profile build is skipped.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Artifact directory.")
    parser.add_argument("--input-name", default="images", help="Model input tensor name.")
    parser.add_argument("--workspace-mib", type=int, default=2048, help="TensorRT workspace pool in MiB.")
    parser.add_argument("--warmup-ms", type=int, default=500, help="trtexec warmup time in milliseconds.")
    parser.add_argument("--duration-sec", type=int, default=5, help="trtexec benchmark duration in seconds.")
    parser.add_argument("--avg-runs", type=int, default=10, help="Iterations averaged per timing sample.")
    parser.add_argument(
        "--skip-dynamic",
        action="store_true",
        help="Build only static FP32 and FP16 engines even if the dynamic ONNX exists.",
    )
    parser.add_argument(
        "--builds",
        nargs="+",
        choices=("static_fp32", "static_fp16", "dynamic_fp16"),
        help="Optional subset of builds to run. Defaults to all available builds.",
    )
    parser.add_argument(
        "--dynamic-min",
        default="1x3x320x320",
        help="Dynamic profile minimum shape without input name.",
    )
    parser.add_argument(
        "--dynamic-opt",
        default="1x3x640x640",
        help="Dynamic profile optimum shape without input name.",
    )
    parser.add_argument(
        "--dynamic-max",
        default="4x3x640x640",
        help="Dynamic profile maximum shape without input name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned trtexec commands without running them.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.workspace_mib <= 0:
        raise ValueError("--workspace-mib must be positive")
    if args.warmup_ms < 0:
        raise ValueError("--warmup-ms cannot be negative")
    if args.duration_sec <= 0:
        raise ValueError("--duration-sec must be positive")
    if args.avg_runs <= 0:
        raise ValueError("--avg-runs must be positive")
    if not args.input_name:
        raise ValueError("--input-name cannot be empty")
    if not args.onnx.exists():
        raise FileNotFoundError(
            f"Static ONNX not found: {args.onnx}\n"
            "Run lesson 05 first: python3 05_torch_to_onnx/export_yolov8_onnx.py"
        )
    if shutil.which("trtexec") is None:
        raise FileNotFoundError("trtexec was not found in PATH. Complete lesson 00 inside the TensorRT container.")


def shape_spec(input_name: str, shape: str) -> str:
    if ":" in shape:
        return shape
    return f"{input_name}:{shape}"


def planned_builds(args: argparse.Namespace) -> list[EngineBuild]:
    output_dir = args.output_dir.resolve()
    static_onnx = args.onnx.resolve()
    dynamic_onnx = args.dynamic_onnx.resolve()

    builds = [
        EngineBuild(
            name="static_fp32",
            onnx_path=static_onnx,
            engine_path=output_dir / "yolov8n_static_fp32.engine",
            log_path=output_dir / "yolov8n_static_fp32.log",
            times_path=output_dir / "yolov8n_static_fp32_times.json",
            layer_info_path=output_dir / "yolov8n_static_fp32_layers.json",
            profile_path=output_dir / "yolov8n_static_fp32_profile.json",
            fp16=False,
            dynamic=False,
        ),
        EngineBuild(
            name="static_fp16",
            onnx_path=static_onnx,
            engine_path=output_dir / "yolov8n_static_fp16.engine",
            log_path=output_dir / "yolov8n_static_fp16.log",
            times_path=output_dir / "yolov8n_static_fp16_times.json",
            layer_info_path=output_dir / "yolov8n_static_fp16_layers.json",
            profile_path=output_dir / "yolov8n_static_fp16_profile.json",
            fp16=True,
            dynamic=False,
        ),
    ]

    if not args.skip_dynamic and dynamic_onnx.exists():
        builds.append(
            EngineBuild(
                name="dynamic_fp16",
                onnx_path=dynamic_onnx,
                engine_path=output_dir / "yolov8n_dynamic_fp16.engine",
                log_path=output_dir / "yolov8n_dynamic_fp16.log",
                times_path=output_dir / "yolov8n_dynamic_fp16_times.json",
                layer_info_path=output_dir / "yolov8n_dynamic_fp16_layers.json",
                profile_path=output_dir / "yolov8n_dynamic_fp16_profile.json",
                fp16=True,
                dynamic=True,
                shapes=shape_spec(args.input_name, args.dynamic_opt),
                min_shapes=shape_spec(args.input_name, args.dynamic_min),
                opt_shapes=shape_spec(args.input_name, args.dynamic_opt),
                max_shapes=shape_spec(args.input_name, args.dynamic_max),
            )
        )

    if args.builds:
        requested = set(args.builds)
        available = {build.name for build in builds}
        missing = sorted(requested - available)
        if missing:
            raise FileNotFoundError(
                "Requested build(s) are not available: "
                + ", ".join(missing)
                + ". Check whether the dynamic ONNX exists or remove --skip-dynamic."
            )
        builds = [build for build in builds if build.name in requested]

    return builds


def trtexec_command(build: EngineBuild, args: argparse.Namespace) -> list[str]:
    command = [
        "trtexec",
        f"--onnx={build.onnx_path}",
        f"--saveEngine={build.engine_path}",
        f"--memPoolSize=workspace:{args.workspace_mib}",
        f"--timingCacheFile={args.output_dir.resolve() / 'trtexec_timing.cache'}",
        "--profilingVerbosity=detailed",
        "--dumpLayerInfo",
        "--dumpProfile",
        "--separateProfileRun",
        f"--exportTimes={build.times_path}",
        f"--exportLayerInfo={build.layer_info_path}",
        f"--exportProfile={build.profile_path}",
        f"--warmUp={args.warmup_ms}",
        f"--duration={args.duration_sec}",
        f"--avgRuns={args.avg_runs}",
        "--percentile=50,90,95,99",
    ]

    if build.fp16:
        command.append("--fp16")
    if build.dynamic:
        command.extend(
            [
                f"--minShapes={build.min_shapes}",
                f"--optShapes={build.opt_shapes}",
                f"--maxShapes={build.max_shapes}",
                f"--shapes={build.shapes}",
            ]
        )

    return command


def run_build(build: EngineBuild, args: argparse.Namespace) -> None:
    command = trtexec_command(build, args)
    print(f"\n== {build.name} ==")
    print(" ".join(command))

    if args.dry_run:
        return

    build.log_path.parent.mkdir(parents=True, exist_ok=True)
    with build.log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"{build.name} failed with exit code {return_code}. See {build.log_path}")
    if not build.engine_path.exists():
        raise RuntimeError(f"{build.name} finished but did not create {build.engine_path}")

    print(f"engine: {build.engine_path}")
    print(f"log: {build.log_path}")


def write_manifest(builds: list[EngineBuild], args: argparse.Namespace) -> None:
    manifest = {
        "static_onnx": str(args.onnx.resolve()),
        "dynamic_onnx": str(args.dynamic_onnx.resolve()),
        "workspace_mib": args.workspace_mib,
        "warmup_ms": args.warmup_ms,
        "duration_sec": args.duration_sec,
        "avg_runs": args.avg_runs,
        "builds": [
            {
                "name": build.name,
                "onnx": str(build.onnx_path),
                "engine": str(build.engine_path),
                "log": str(build.log_path),
                "times": str(build.times_path),
                "layers": str(build.layer_info_path),
                "profile": str(build.profile_path),
                "fp16": build.fp16,
                "dynamic": build.dynamic,
                "shapes": build.shapes,
                "min_shapes": build.min_shapes,
                "opt_shapes": build.opt_shapes,
                "max_shapes": build.max_shapes,
            }
            for build in builds
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.onnx = args.onnx.resolve()
    args.dynamic_onnx = args.dynamic_onnx.resolve()
    args.output_dir = args.output_dir.resolve()

    try:
        validate_args(args)
        builds = planned_builds(args)
        if not args.skip_dynamic and not args.dynamic_onnx.exists():
            print(
                f"Dynamic ONNX not found, skipping dynamic profile build: {args.dynamic_onnx}\n"
                "Create it with: python3 05_torch_to_onnx/export_yolov8_onnx.py --dynamic "
                "--output 05_torch_to_onnx/outputs/yolov8n_dynamic.onnx"
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for build in builds:
            run_build(build, args)
        write_manifest(builds, args)

        if not args.dry_run:
            print(f"\nmanifest: {args.output_dir / 'build_manifest.json'}")
            print("Next: python3 06_trtexec_engine/summarize_results.py")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
