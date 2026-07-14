#!/usr/bin/env python3
"""Numerically validate the rewritten graph against a NumPy reference."""

from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent


def main() -> int:
    model = ROOT / "outputs/rewritten_swish.onnx"
    values = np.array([[-2.0, -0.5, 0.5, 3.0]], dtype=np.float32)
    biased = values + 0.25
    reference = biased * (1.0 / (1.0 + np.exp(-biased))) * 1.5
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    output = session.run(None, {"input": values})[0]
    np.testing.assert_allclose(output, reference, rtol=1e-6, atol=1e-6)
    print(f"max_abs={np.max(np.abs(output - reference)):.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
