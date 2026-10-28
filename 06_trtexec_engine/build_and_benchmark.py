#!/usr/bin/env python3
"""Build and benchmark TensorRT engines with validated inputs and reusable caches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "06_trtexec_engine/outputs"
FP32 = ROOT / "05_torch_to_onnx/outputs/yolov8n.onnx"
DYN = ROOT / "05_torch_to_onnx/outputs/yolov8n_dynamic.onnx"
AUTO = OUT / "yolov8n_static_autocast_fp16.onnx"
DYNAUTO = OUT / "yolov8n_dynamic_autocast_fp16.onnx"
BUILD_NAMES = ("static_fp32", "static_fp16_legacy", "dynamic_fp16_legacy",
               "static_fp16_strong", "dynamic_fp16_strong")


@dataclass(frozen=True)
class EngineBuild:
    name: str
    onnx_path: Path
    source_path: Path
    engine_path: Path
    log_path: Path
    times_path: Path
    layer_info_path: Path
    profile_path: Path
    precision: str
    typing_mode: str
    deprecated: bool
    validation_report: Path | None
    dynamic: bool
    shapes: str | None = None
    min_shapes: str | None = None
    opt_shapes: str | None = None
    max_shapes: str | None = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=FP32)
    parser.add_argument("--dynamic-onnx", type=Path, default=DYN)
    parser.add_argument("--static-autocast-onnx", type=Path, default=AUTO)
    parser.add_argument("--dynamic-autocast-onnx", type=Path, default=DYNAUTO)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--input-name", default="images")
    parser.add_argument("--workspace-mib", type=int, default=2048)
    parser.add_argument("--warmup-ms", type=int, default=500)
    parser.add_argument("--duration-sec", type=int, default=5)
    parser.add_argument("--avg-runs", type=int, default=10)
    parser.add_argument("--skip-dynamic", action="store_true")
    parser.add_argument("--skip-strongly-typed", action="store_true")
    parser.add_argument("--builds", nargs="+", choices=BUILD_NAMES)
    parser.add_argument("--dynamic-min", default="1x3x320x320")
    parser.add_argument("--dynamic-opt", default="1x3x640x640")
    parser.add_argument("--dynamic-max", default="4x3x640x640")
    parser.add_argument("--timing-cache-file", type=Path,
                        help="Reuse a timing cache from this same GPU/software environment.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild even when a matching engine receipt exists; still use timing cache.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def timing_cache(args) -> Path:
    return (args.timing_cache_file or args.output_dir / "trtexec_timing.cache").resolve()


def shape_spec(name, shape):
    return shape if ":" in shape else f"{name}:{shape}"


def validate(args):
    if (args.workspace_mib <= 0 or args.duration_sec <= 0 or
            args.avg_runs <= 0 or args.warmup_ms < 0):
        raise ValueError("workspace, duration and avg-runs must be positive; warmup must be non-negative")
    if shutil.which("trtexec") is None:
        raise FileNotFoundError("trtexec was not found in PATH")


def validate_conversion(build: EngineBuild):
    """A leftover ONNX from a failed conversion must never bypass its numerical gate."""
    if build.validation_report is None:
        return
    if not build.validation_report.is_file():
        raise FileNotFoundError(f"Validation report not found: {build.validation_report}; rerun prepare_fp16_onnx.py")
    report = json.loads(build.validation_report.read_text(encoding="utf-8"))
    comparison = report.get("comparison") if isinstance(report, dict) else None
    if not isinstance(comparison, dict) or comparison.get("passed") is not True:
        raise ValueError(f"FP16 conversion validation did not pass: {build.validation_report}")
    for field, path in (("source", build.source_path), ("converted", build.onnx_path)):
        if report.get(f"{field}_sha256") != sha256(path):
            raise ValueError(f"{field} ONNX hash differs from validation report: {path}; rerun prepare_fp16_onnx.py")


def planned(args):
    output_dir = args.output_dir.resolve()
    requested = set(args.builds or BUILD_NAMES)
    disabled = set()
    if args.skip_dynamic:
        disabled.update(name for name in BUILD_NAMES if name.startswith("dynamic_"))
    if args.skip_strongly_typed:
        disabled.update(name for name in BUILD_NAMES if name.endswith("_strong"))
    if args.builds and requested & disabled:
        raise ValueError("Requested builds are disabled: " + ", ".join(sorted(requested & disabled)))

    builds = []
    for name in BUILD_NAMES:
        if name not in requested or name in disabled:
            continue
        dynamic = name.startswith("dynamic_")
        strong = name.endswith("_strong")
        source = (args.dynamic_onnx if dynamic else args.onnx).resolve()
        onnx = ((args.dynamic_autocast_onnx if dynamic else args.static_autocast_onnx)
                if strong else source).resolve()
        for path in (source, onnx):
            if not path.is_file():
                raise FileNotFoundError(f"ONNX not found: {path}; run lesson 05 and prepare_fp16_onnx.py as appropriate")
        stem = f"yolov8n_{name}"
        kind = "dynamic" if dynamic else "static"
        build = EngineBuild(
            name=name, onnx_path=onnx, source_path=source,
            engine_path=output_dir / f"{stem}.engine", log_path=output_dir / f"{stem}.log",
            times_path=output_dir / f"{stem}_times.json",
            layer_info_path=output_dir / f"{stem}_layers.json",
            profile_path=output_dir / f"{stem}_profile.json",
            precision="fp32" if name == "static_fp32" else "fp16",
            typing_mode="strongly_typed" if strong else "weakly_typed",
            deprecated=name.endswith("_legacy"),
            # Reports belong next to the converted graph, not to the engine output directory.
            validation_report=onnx.parent / f"{kind}_fp16_onnx_validation.json" if strong else None,
            dynamic=dynamic,
            shapes=shape_spec(args.input_name, args.dynamic_opt) if dynamic else None,
            min_shapes=shape_spec(args.input_name, args.dynamic_min) if dynamic else None,
            opt_shapes=shape_spec(args.input_name, args.dynamic_opt) if dynamic else None,
            max_shapes=shape_spec(args.input_name, args.dynamic_max) if dynamic else None,
        )
        validate_conversion(build)
        builds.append(build)
    return builds


def command(build, args, reuse_engine=False):
    cmd = ["trtexec"]
    if reuse_engine:
        cmd.append(f"--loadEngine={build.engine_path}")
    else:
        cmd.extend([
            f"--onnx={build.onnx_path}", f"--saveEngine={build.engine_path}",
            f"--memPoolSize=workspace:{args.workspace_mib}",
            f"--timingCacheFile={timing_cache(args)}", "--profilingVerbosity=detailed",
        ])
        cmd.append("--noTF32" if build.precision == "fp32" else
                   "--stronglyTyped" if build.typing_mode == "strongly_typed" else "--fp16")
        if build.dynamic:
            cmd.extend([f"--minShapes={build.min_shapes}", f"--optShapes={build.opt_shapes}",
                        f"--maxShapes={build.max_shapes}"])
    cmd.extend([
        "--dumpLayerInfo", "--dumpProfile", "--separateProfileRun",
        f"--exportTimes={build.times_path}", f"--exportLayerInfo={build.layer_info_path}",
        f"--exportProfile={build.profile_path}", f"--warmUp={args.warmup_ms}",
        f"--duration={args.duration_sec}", f"--avgRuns={args.avg_runs}", "--percentile=50,90,95,99",
    ])
    if build.dynamic:
        cmd.append(f"--shapes={build.shapes}")
    return cmd


def engine_signature(build, args, environment):
    # Benchmark duration and timing-cache location do not change the serialized engine.
    return {"schema": 1, "onnx_sha256": sha256(build.onnx_path),
            "precision": build.precision, "typing_mode": build.typing_mode,
            "workspace_mib": args.workspace_mib,
            "min_shapes": build.min_shapes, "opt_shapes": build.opt_shapes,
            "max_shapes": build.max_shapes, "environment": environment}


def reusable_engine(build, signature):
    receipt_path = build.engine_path.with_suffix(".build.json")
    if not receipt_path.is_file() or not build.engine_path.is_file():
        return False
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            return False
        return (receipt["signature"] == signature and build.engine_path.stat().st_size > 0
                and receipt["engine_sha256"] == sha256(build.engine_path))
    except (ValueError, KeyError, OSError):
        return False


def run(build, args, environment):
    validate_conversion(build)
    signature = engine_signature(build, args, environment)
    reuse = not args.rebuild and reusable_engine(build, signature)
    cmd = command(build, args, reuse)
    print(f"\n== {build.name} ({'reuse engine' if reuse else 'build with timing cache'}) ==\n{shlex.join(cmd)}", flush=True)
    build.log_path.parent.mkdir(parents=True, exist_ok=True)
    timing_cache(args).parent.mkdir(parents=True, exist_ok=True)
    # Remove old evidence before execution so a failed run cannot inherit successful exports.
    for path in (build.times_path, build.layer_info_path, build.profile_path):
        path.unlink(missing_ok=True)
    if not reuse:
        build.engine_path.with_suffix(".build.json").unlink(missing_ok=True)
        build.engine_path.unlink(missing_ok=True)
    with build.log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    log_text = build.log_path.read_text(encoding="utf-8", errors="replace")
    if result.returncode or "&&&& PASSED TensorRT.trtexec" not in log_text:
        raise RuntimeError(f"{build.name} failed; see {build.log_path}")
    for path in (build.engine_path, build.times_path, build.layer_info_path, build.profile_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty artifact: {path}")
    for path in (build.times_path, build.layer_info_path, build.profile_path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if path == build.layer_info_path:
            valid = isinstance(payload, dict) and bool(payload.get("Layers"))
        else:
            valid = isinstance(payload, list) and bool(payload)
        if not valid:
            raise ValueError(f"JSON export contains no evidence: {path}")
    if not reuse:
        receipt = {"signature": signature, "engine_sha256": sha256(build.engine_path),
                   "built_at_utc": datetime.now(timezone.utc).isoformat(), "command": cmd}
        build.engine_path.with_suffix(".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return {"command": cmd, "reused_engine": reuse,
            "measured_at_utc": datetime.now(timezone.utc).isoformat()}


def env():
    import tensorrt as trt

    def output(cmd):
        return subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, check=True).stdout.strip()

    return {
        "course_image": "learn-tensorrt:25.11 (nvcr.io/nvidia/pytorch:25.11-py3)",
        "nvidia_container_release": os.environ.get("NVIDIA_PYTORCH_VERSION", "unknown"),
        "nvidia_build_id": os.environ.get("NVIDIA_BUILD_ID", "unknown"),
        "container_hostname": platform.node(), "python": platform.python_version(),
        "tensorrt": trt.__version__, "cuda_toolkit": output(["nvcc", "--version"]),
        "gpu_and_driver": output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap",
                                  "--format=csv,noheader"]),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
    }


def manifest(builds, args, environment, runs):
    import onnx

    records = []
    for build, run_info in zip(builds, runs):
        model = onnx.load(build.onnx_path, load_external_data=False)
        input_shapes = {}
        for value in model.graph.input:
            input_shapes[value.name] = [
                dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param or "?"
                for dimension in value.type.tensor_type.shape.dim
            ]
        record = {
            "name": build.name, "precision": build.precision, "typing_mode": build.typing_mode,
            "deprecated": build.deprecated, "dynamic": build.dynamic,
            "fp16": build.precision == "fp16", "onnx": str(build.onnx_path),
            "onnx_sha256": sha256(build.onnx_path), "engine": str(build.engine_path),
            "log": str(build.log_path), "times": str(build.times_path),
            "layers": str(build.layer_info_path), "profile": str(build.profile_path),
            "validation_report": str(build.validation_report) if build.validation_report else None,
            "tf32_policy": "disabled" if build.precision == "fp32" else "TensorRT default",
            "shapes": build.shapes, "min_shapes": build.min_shapes,
            "onnx_input_shapes": input_shapes,
            "opt_shapes": build.opt_shapes, "max_shapes": build.max_shapes,
            **run_info,
        }
        records.append(record)
    payload = {"runtime_environment": environment, "static_onnx": str(args.onnx.resolve()),
               "dynamic_onnx": str(args.dynamic_onnx.resolve()), "workspace_mib": args.workspace_mib,
               "warmup_ms": args.warmup_ms, "duration_sec": args.duration_sec,
               "avg_runs": args.avg_runs, "timing_cache": str(timing_cache(args)), "builds": records}
    (args.output_dir / "build_manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    try:
        validate(args)
        builds = planned(args)  # Validate every selected graph before starting expensive work.
        if args.dry_run:
            for build in builds:
                print(f"== {build.name} ==\n{shlex.join(command(build, args))}")
            return 0
        environment = env()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "build_manifest.json").unlink(missing_ok=True)
        runs = [run(build, args, environment) for build in builds]
        manifest(builds, args, environment, runs)
        print(f"manifest: {args.output_dir / 'build_manifest.json'}")
        return 0
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
