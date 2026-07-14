#!/usr/bin/env python3
from pathlib import Path

import onnx
from onnx import TensorProto, helper

ROOT = Path(__file__).resolve().parent
output = ROOT / "outputs/scale_shift.onnx"
output.parent.mkdir(parents=True, exist_ok=True)
x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
node = helper.make_node("ScaleShift", ["input"], ["output"], name="scale_shift",
                        domain="com.acme", scale=2.0, shift=-1.0)
graph = helper.make_graph([node], "scale_shift_plugin_graph", [x], [y])
model = helper.make_model(graph, producer_name="lesson19a",
                          opset_imports=[helper.make_opsetid("", 17),
                                         helper.make_opsetid("com.acme", 1)])
onnx.save(model, output)
print(f"wrote {output}")
