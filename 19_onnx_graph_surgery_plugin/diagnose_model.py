#!/usr/bin/env python3
"""List custom-domain nodes before escalating to rewrite, surgery, or plugin work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx

ROOT = Path(__file__).resolve().parent


def diagnose(path: Path) -> dict:
    model = onnx.load(path)
    custom = [{"name": node.name, "op_type": node.op_type, "domain": node.domain}
              for node in model.graph.node if node.domain not in ("", "ai.onnx")]
    return {"model": str(path), "node_count": len(model.graph.node),
            "custom_domain_nodes": custom,
            "recommended_escalation": (
                "replace with standard ONNX operators before writing a TensorRT plugin"
                if custom else "run TensorRT parser and inspect any remaining unsupported operators")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", type=Path,
                        default=ROOT / "outputs/unsupported_swish.onnx")
    args = parser.parse_args()
    print(json.dumps(diagnose(args.model), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
