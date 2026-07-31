#!/usr/bin/env python3
"""Replace com.acme::AcmeSwish with standard ONNX Sigmoid and Mul nodes."""

from pathlib import Path

import onnx
import onnx_graphsurgeon as gs

ROOT = Path(__file__).resolve().parent


def rewrite(source: Path, destination: Path) -> int:
    graph = gs.import_onnx(onnx.load(source))
    replacements = 0
    for node in list(graph.nodes):
        if node.domain == "com.acme" and node.op == "AcmeSwish":
            if len(node.inputs) != 1 or len(node.outputs) != 1:
                raise ValueError("AcmeSwish must have one input and one output")
            source_tensor = node.inputs[0]
            original_output = node.outputs[0]
            sigmoid = gs.Variable(node.name + "_sigmoid", dtype=source_tensor.dtype,
                                  shape=source_tensor.shape)
            node.outputs.clear()
            graph.nodes.extend([
                gs.Node(op="Sigmoid", name=node.name + "_sigmoid_node",
                        inputs=[source_tensor], outputs=[sigmoid]),
                gs.Node(op="Mul", name=node.name + "_mul_node",
                        inputs=[source_tensor, sigmoid], outputs=[original_output]),
            ])
            replacements += 1
    if replacements != 1:
        raise RuntimeError(f"expected one AcmeSwish node, replaced {replacements}")
    graph.cleanup().toposort()
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(gs.export_onnx(graph), destination)
    onnx.checker.check_model(onnx.load(destination))
    return replacements


def main() -> int:
    source = ROOT / "outputs/unsupported_swish.onnx"
    destination = ROOT / "outputs/rewritten_swish.onnx"
    count = rewrite(source, destination)
    print(f"replaced {count} node and wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
