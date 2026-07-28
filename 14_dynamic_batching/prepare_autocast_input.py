#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    if args.batch <= 0:
        parser.error("batch must be positive")
    tensor = np.load(args.input)
    if tensor.shape != (1, 3, 640, 640) or tensor.dtype != np.float32:
        raise ValueError(
            f"expected float32 [1,3,640,640] input, got {tensor.dtype} {tensor.shape}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, images=np.repeat(tensor, args.batch, axis=0))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
