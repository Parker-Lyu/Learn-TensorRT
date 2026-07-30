#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class Result(ctypes.Structure):
    _fields_ = [("batch_size", ctypes.c_size_t), ("output_elements", ctypes.c_size_t),
                ("h2d_ms", ctypes.c_float), ("compute_ms", ctypes.c_float),
                ("d2h_ms", ctypes.c_float), ("output_checksum", ctypes.c_double)]


class TensorRtSession:
    def __init__(self, library: Path, engine: Path):
        self.lib = ctypes.CDLL(str(library))
        self.lib.trt_session_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.trt_session_create.restype = ctypes.c_int
        self.lib.trt_session_destroy.argtypes = [ctypes.c_void_p]
        self.lib.trt_session_infer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(Result)]
        self.lib.trt_session_infer.restype = ctypes.c_int
        self.lib.trt_last_error.restype = ctypes.c_char_p
        self.handle = ctypes.c_void_p()
        self._check(self.lib.trt_session_create(str(engine).encode(), ctypes.byref(self.handle)))

    def _check(self, code: int):
        if code:
            raise RuntimeError(self.lib.trt_last_error().decode())

    def infer(self, tensor: np.ndarray) -> Result:
        array = np.ascontiguousarray(tensor, dtype=np.float32)
        if array.ndim != 4:
            raise ValueError("input must be NCHW")
        result = Result()
        self._check(self.lib.trt_session_infer(self.handle,
            array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), array.size,
            array.shape[0], ctypes.byref(result)))
        return result

    def close(self):
        if self.handle:
            self.lib.trt_session_destroy(self.handle)
            self.handle = None

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path,
                        default=ROOT / "28_cpp_shared_library_python_binding/build/libtrt_inference.so")
    parser.add_argument("--engine", type=Path,
                        default=ROOT / "17_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine")
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()
    tensor = np.full((args.batch, 3, 640, 640), 0.5, np.float32)
    with TensorRtSession(args.library, args.engine) as session:
        result = session.infer(tensor)
    print(f"batch={result.batch_size} output_elements={result.output_elements} "
          f"compute_ms={result.compute_ms:.3f} checksum={result.output_checksum:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
