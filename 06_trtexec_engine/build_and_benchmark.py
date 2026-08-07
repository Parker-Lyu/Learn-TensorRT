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
    o=a.output_dir.resolve(); s=a.onnx.resolve(); d=a.dynamic_onnx.resolve()
    def make(name,path,dynamic,mode,deprecated=False,report=None):
        stem=f"yolov8n_{name}"; return EngineBuild(name,path.resolve(),o/f"{stem}.engine",o/f"{stem}.log",o/f"{stem}_times.json",o/f"{stem}_layers.json",o/f"{stem}_profile.json","fp32" if mode=="none" else "fp16",mode,deprecated,report,dynamic,
            shape_spec(a.input_name,a.dynamic_opt) if dynamic else None,shape_spec(a.input_name,a.dynamic_min) if dynamic else None,shape_spec(a.input_name,a.dynamic_opt) if dynamic else None,shape_spec(a.input_name,a.dynamic_max) if dynamic else None)
    b=[make("static_fp32",s,False,"none"),make("static_fp16_legacy",s,False,"weakly_typed",True)]
    if not a.skip_dynamic and d.exists(): b.append(make("dynamic_fp16_legacy",d,True,"weakly_typed",True))
    if not a.skip_strongly_typed:
        if not a.static_autocast_onnx.exists(): raise FileNotFoundError(f"{a.static_autocast_onnx} not found; run prepare_fp16_onnx.py first")
        b.append(make("static_fp16_strong",a.static_autocast_onnx,False,"strongly_typed",False,o/"static_fp16_onnx_validation.json"))
        if not a.skip_dynamic and a.dynamic_autocast_onnx.exists(): b.append(make("dynamic_fp16_strong",a.dynamic_autocast_onnx,True,"strongly_typed",False,o/"dynamic_fp16_onnx_validation.json"))
    if a.builds:
        names=set(a.builds); missing=names-{x.name for x in b}
        if missing: raise FileNotFoundError("Requested builds unavailable: "+", ".join(sorted(missing)))
        b=[x for x in b if x.name in names]
    return b

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
    return {"python":platform.python_version(),"tensorrt":trt.__version__,"cuda_toolkit":os.environ.get("CUDA_VERSION",out(["nvcc","--version"])),"gpu_and_driver":out(["nvidia-smi","--query-gpu=name,driver_version,compute_cap","--format=csv,noheader"])}
def manifest(bs,a):
    a.output_dir.mkdir(parents=True,exist_ok=True); (a.output_dir/"build_manifest.json").write_text(json.dumps({"runtime_environment":env(),"builds":[{**{k:getattr(b,k) for k in ("name","precision","typing_mode","deprecated","dynamic")},"onnx":str(b.onnx_path),"engine":str(b.engine_path),"validation_report":str(b.validation_report) if b.validation_report else None} for b in bs]},indent=2)+"\n")
def main():
    a=parse_args(); a.output_dir=a.output_dir.resolve()
    try:
        validate(a); bs=planned(a)
        for b in bs: run(b,a)
        if not a.dry_run: manifest(bs,a); print(f"manifest: {a.output_dir/'build_manifest.json'}")
        return 0
    except Exception as e: print(f"error: {e}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
