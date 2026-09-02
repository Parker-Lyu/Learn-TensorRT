#!/usr/bin/env python3
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=Path, default=ROOT / "29_cpp_shared_library_python_binding/build")
    parser.add_argument("--engine", type=Path, default=ROOT / "17_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine")
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module))
    import trt_inference_py
    tensor = np.full((args.batch, 3, 640, 640), 0.5, dtype=np.float32)
    result = trt_inference_py.TensorRtSession(str(args.engine)).infer(tensor)
    print(f"batch={result['batch_size']} output_elements={result['output_elements']} compute_ms={result['compute_ms']:.3f} checksum={result['output_checksum']:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
