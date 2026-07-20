#!/usr/bin/env python3
"""Focused tests for the TensorRT 8.6 ModelOpt Q/DQ build wrapper."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "build_trt86_qdq_engine.py"
SPEC = importlib.util.spec_from_file_location("build_trt86_qdq_engine", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class BuildTrt86QdqEngineTests(unittest.TestCase):
    def test_command_enables_int8_and_fp16_for_explicit_qdq(self) -> None:
        command = BUILD.build_command(
            "trtexec",
            Path("model.onnx"),
            Path("model.engine"),
            Path("layers.json"),
            Path("timing.cache"),
        )
        self.assertIn("--int8", command)
        self.assertIn("--fp16", command)
        self.assertIn("--builderOptimizationLevel=3", command)
        self.assertNotIn("--stronglyTyped", command)
        self.assertFalse(any(argument.startswith("--calib=") for argument in command))

    def test_precision_evidence_requires_both_int8_and_fp16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layers.json"
            path.write_text("Format/Datatype: Int8 Format/Datatype: FP16", encoding="utf-8")
            evidence = BUILD.precision_evidence(path)
            self.assertGreater(evidence["int8_mentions"], 0)
            self.assertGreater(evidence["fp16_mentions"], 0)

            path.write_text("Format/Datatype: Int8", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no FP16"):
                BUILD.precision_evidence(path)


if __name__ == "__main__":
    unittest.main()

