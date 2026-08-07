#!/usr/bin/env python3
"""Build reproducible TensorRT engines for the legacy and modern FP16 routes."""
from __future__ import annotations
import argparse, json, os, platform, shutil, subprocess, sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "06_trtexec_engine" / "outputs"
FP32 = ROOT / "05_torch_to_onnx/outputs/yolov8n.onnx"
DYN = ROOT / "05_torch_to_onnx/outputs/yolov8n_dynamic.onnx"
AUTO = OUT / "yolov8n_static_autocast_fp16.onnx"
DYNAUTO = OUT / "yolov8n_dynamic_autocast_fp16.onnx"

@dataclass(frozen=True)
class EngineBuild:
    name: str; onnx_path: Path; engine_path: Path; log_path: Path; times_path: Path
    layer_info_path: Path; profile_path: Path; precision: str; typing_mode: str
    deprecated: bool; validation_report: Path | None; dynamic: bool
    shapes: str | None = None; min_shapes: str | None = None; opt_shapes: str | None = None; max_shapes: str | None = None

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx",type=Path,default=FP32); p.add_argument("--dynamic-onnx",type=Path,default=DYN)
    p.add_argument("--static-autocast-onnx",type=Path,default=AUTO); p.add_argument("--dynamic-autocast-onnx",type=Path,default=DYNAUTO)
    p.add_argument("--output-dir",type=Path,default=OUT); p.add_argument("--input-name",default="images")
    p.add_argument("--workspace-mib",type=int,default=2048); p.add_argument("--warmup-ms",type=int,default=500)
    p.add_argument("--duration-sec",type=int,default=5); p.add_argument("--avg-runs",type=int,default=10)
    p.add_argument("--skip-dynamic",action="store_true"); p.add_argument("--skip-strongly-typed",action="store_true")
    p.add_argument("--builds",nargs="+",choices=("static_fp32","static_fp16_legacy","dynamic_fp16_legacy","static_fp16_strong","dynamic_fp16_strong"))
    p.add_argument("--dynamic-min",default="1x3x320x320"); p.add_argument("--dynamic-opt",default="1x3x640x640"); p.add_argument("--dynamic-max",default="4x3x640x640")
    p.add_argument("--dry-run",action="store_true"); return p.parse_args()

def shape_spec(name, shape): return shape if ":" in shape else f"{name}:{shape}"
def validate(a):
    if a.workspace_mib<=0 or a.duration_sec<=0 or a.avg_runs<=0 or a.warmup_ms<0: raise ValueError("invalid build settings")
    if not a.onnx.exists(): raise FileNotFoundError(f"Static ONNX not found: {a.onnx}; run lesson 05 first")
    if shutil.which("trtexec") is None: raise FileNotFoundError("trtexec was not found in PATH")

def planned(a):
    output_dir = a.output_dir.resolve()
    static_onnx = a.onnx.resolve()
    dynamic_onnx = a.dynamic_onnx.resolve()
    requested = set(a.builds or ())

    def wants(name):
        return not requested or name in requested

    def make(name, path, dynamic, mode, deprecated=False, report=None):
        stem = f"yolov8n_{name}"
        return EngineBuild(
            name, path.resolve(), output_dir / f"{stem}.engine", output_dir / f"{stem}.log",
            output_dir / f"{stem}_times.json", output_dir / f"{stem}_layers.json",
            output_dir / f"{stem}_profile.json", "fp32" if mode == "none" else "fp16", mode,
            deprecated, report, dynamic,
            shape_spec(a.input_name, a.dynamic_opt) if dynamic else None,
            shape_spec(a.input_name, a.dynamic_min) if dynamic else None,
            shape_spec(a.input_name, a.dynamic_opt) if dynamic else None,
            shape_spec(a.input_name, a.dynamic_max) if dynamic else None,
        )

    builds = []
    if wants("static_fp32"):
        builds.append(make("static_fp32", static_onnx, False, "none"))
    if wants("static_fp16_legacy"):
        builds.append(make("static_fp16_legacy", static_onnx, False, "weakly_typed", True))

    dynamic_legacy_requested = wants("dynamic_fp16_legacy") and not a.skip_dynamic
    if dynamic_legacy_requested:
        if not dynamic_onnx.is_file():
            raise FileNotFoundError(f"Dynamic ONNX not found: {dynamic_onnx}; run lesson 05 dynamic export first")
        builds.append(make("dynamic_fp16_legacy", dynamic_onnx, True, "weakly_typed", True))

    if not a.skip_strongly_typed:
        if wants("static_fp16_strong"):
            if not a.static_autocast_onnx.is_file():
                raise FileNotFoundError(f"{a.static_autocast_onnx} not found; run prepare_fp16_onnx.py first")
            builds.append(make("static_fp16_strong", a.static_autocast_onnx, False, "strongly_typed", False, output_dir / "static_fp16_onnx_validation.json"))
        if wants("dynamic_fp16_strong") and not a.skip_dynamic:
            if not a.dynamic_autocast_onnx.is_file():
                raise FileNotFoundError(f"{a.dynamic_autocast_onnx} not found; run prepare_fp16_onnx.py --models dynamic first")
            builds.append(make("dynamic_fp16_strong", a.dynamic_autocast_onnx, True, "strongly_typed", False, output_dir / "dynamic_fp16_onnx_validation.json"))

    if requested:
        available = {build.name for build in builds}
        missing = requested - available
        if missing:
            raise ValueError("Requested builds are disabled or unavailable: " + ", ".join(sorted(missing)))
    return builds

def command(build,a):
    c=["trtexec",f"--onnx={build.onnx_path}",f"--saveEngine={build.engine_path}",f"--memPoolSize=workspace:{a.workspace_mib}",f"--timingCacheFile={a.output_dir.resolve()/'trtexec_timing.cache'}","--profilingVerbosity=detailed","--dumpLayerInfo","--dumpProfile","--separateProfileRun",f"--exportTimes={build.times_path}",f"--exportLayerInfo={build.layer_info_path}",f"--exportProfile={build.profile_path}",f"--warmUp={a.warmup_ms}",f"--duration={a.duration_sec}",f"--avgRuns={a.avg_runs}","--percentile=50,90,95,99"]
    c.append("--noTF32" if build.typing_mode=="none" else "--fp16" if build.typing_mode=="weakly_typed" else "--stronglyTyped")
    if build.dynamic: c += [f"--minShapes={build.min_shapes}",f"--optShapes={build.opt_shapes}",f"--maxShapes={build.max_shapes}",f"--shapes={build.shapes}"]
    return c

def run(build,a):
    c=command(build,a); print(f"\n== {build.name} ==\n{' '.join(c)}")
    if a.dry_run:return
    build.log_path.parent.mkdir(parents=True,exist_ok=True)
    with build.log_path.open("w") as log:
        r=subprocess.run(c,stdout=log,stderr=subprocess.STDOUT,check=False)
    if r.returncode or not build.engine_path.exists(): raise RuntimeError(f"{build.name} failed; see {build.log_path}")

def env():
    import tensorrt as trt
    def out(c): return subprocess.run(c,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False).stdout.strip()
    return {"course_image":os.environ.get("NVIDIA_PYTORCH_VERSION","unknown"),"nvidia_container_release":os.environ.get("NVIDIA_PYTORCH_VERSION","unknown"),"nvidia_build_id":os.environ.get("NVIDIA_BUILD_ID","unknown"),"python":platform.python_version(),"tensorrt":trt.__version__,"cuda_toolkit":os.environ.get("CUDA_VERSION",out(["nvcc","--version"])),"gpu_and_driver":out(["nvidia-smi","--query-gpu=name,driver_version,compute_cap","--format=csv,noheader"])}
def manifest(bs,a):
    a.output_dir.mkdir(parents=True,exist_ok=True)
    builds=[]
    for b in bs:
        builds.append({"name":b.name,"precision":b.precision,"typing_mode":b.typing_mode,"deprecated":b.deprecated,"dynamic":b.dynamic,"fp16":b.precision=="fp16","onnx":str(b.onnx_path),"engine":str(b.engine_path),"log":str(b.log_path),"times":str(b.times_path),"layers":str(b.layer_info_path),"profile":str(b.profile_path),"validation_report":str(b.validation_report) if b.validation_report else None})
    payload={"runtime_environment":env(),"static_onnx":str(a.onnx.resolve()),"dynamic_onnx":str(a.dynamic_onnx.resolve()),"workspace_mib":a.workspace_mib,"warmup_ms":a.warmup_ms,"duration_sec":a.duration_sec,"avg_runs":a.avg_runs,"builds":builds}
    (a.output_dir/"build_manifest.json").write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
def main():
    a=parse_args(); a.output_dir=a.output_dir.resolve()
    try:
        validate(a); bs=planned(a)
        for b in bs: run(b,a)
        if not a.dry_run: manifest(bs,a); print(f"manifest: {a.output_dir/'build_manifest.json'}")
        return 0
    except Exception as e: print(f"error: {e}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
