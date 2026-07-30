#!/usr/bin/env python3
"""Create a small ONNX model containing an intentionally unsupported custom Swish node."""

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parent


def main() -> int:
    output = ROOT / "outputs/unsupported_swish.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    bias = numpy_helper.from_array(np.full((1, 4), 0.25, np.float32), "bias")
    scale = numpy_helper.from_array(np.array(1.5, np.float32), "scale")
    nodes = [
        helper.make_node("Add", ["input", "bias"], ["biased"], name="add_bias"),
        helper.make_node("AcmeSwish", ["biased"], ["swish"], name="unsupported_swish",
                         domain="com.acme"),
        helper.make_node("Mul", ["swish", "scale"], ["output"], name="scale_output"),
    ]
    graph = helper.make_graph(nodes, "unsupported_swish_demo", [x], [y], [bias, scale])
    model = helper.make_model(
        graph, producer_name="lesson24",
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.acme", 1)])
    onnx.save(model, output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
