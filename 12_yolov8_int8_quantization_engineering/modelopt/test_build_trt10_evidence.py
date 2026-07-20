#!/usr/bin/env python3
"""Focused tests for TensorRT 10 build orchestration and identity validation."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "build_trt10_evidence.py"
SPEC = importlib.util.spec_from_file_location("build_trt10_evidence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BUILD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUILD
SPEC.loader.exec_module(BUILD)


class BuildTrt10EvidenceTests(unittest.TestCase):
    def test_commands_preserve_each_precision_contract(self) -> None:
        specs = BUILD.build_specs(Path("out"))
        commands = {spec.name: BUILD.build_command("trtexec", spec) for spec in specs}
        self.assertIn("--stronglyTyped", commands["tensorrt_fp32"])
        self.assertNotIn("--fp16", commands["tensorrt_fp32"])
        self.assertIn("--fp16", commands["tensorrt_fp16"])
        self.assertNotIn("--stronglyTyped", commands["tensorrt_fp16"])
        self.assertIn("--stronglyTyped", commands["tensorrt_int8"])
        self.assertNotIn("--fp16", commands["tensorrt_int8"])
        self.assertNotIn("--int8", commands["tensorrt_int8"])
        for command in commands.values():
            self.assertIn("--builderOptimizationLevel=3", command)
            self.assertFalse(any(item.startswith("--calib=") for item in command))

    def test_identity_validation_rejects_changed_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            path.write_bytes(b"expected")
            BUILD.validate_identity(path, BUILD.sha256(path))
            with self.assertRaisesRegex(ValueError, "identity changed"):
                BUILD.validate_identity(path, "0" * 64)

    def test_build_duration_parser_handles_missing_and_present_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "build.log"
            path.write_text("Engine built in 12.345 sec.", encoding="utf-8")
            self.assertEqual(BUILD.log_build_duration(path), 12.345)
            path.write_text("no duration", encoding="utf-8")
            self.assertIsNone(BUILD.log_build_duration(path))


if __name__ == "__main__":
    unittest.main()

