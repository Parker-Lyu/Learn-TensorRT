#!/usr/bin/env python3
"""Focused tests for matched TensorRT 10 performance reporting."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "benchmark_engines.py"
SPEC = importlib.util.spec_from_file_location("benchmark_engines", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkTrt10EvidenceTests(unittest.TestCase):
    def test_command_fixes_matched_methodology(self) -> None:
        command = BENCHMARK.benchmark_command(
            "trtexec", Path("model.engine"), Path("times.json"), 500, 120
        )
        self.assertIn("--warmUp=500", command)
        self.assertIn("--duration=0", command)
        self.assertIn("--iterations=120", command)
        self.assertIn("--infStreams=1", command)

    def test_summary_and_throughput_use_measured_evidence(self) -> None:
        samples = [
            {"latencyMs": index + 1, "computeMs": 1, "h2dMs": 2, "d2hMs": 3}
            for index in range(120)
        ]
        summary = BENCHMARK.summarize(samples)
        self.assertEqual(summary["sample_count"], 120)
        self.assertEqual(summary["latency_ms"]["p90"], 108.0)
        self.assertEqual(summary["gpu_compute_ms"]["mean"], 1.0)
        self.assertEqual(
            BENCHMARK.parse_throughput("Throughput: 1 qps\nThroughput: 700.5 qps"), 700.5
        )

    def test_evaluation_gate_controls_int8_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engines = {}
            artifacts = {}
            for name in ("fp32", "fp16", "int8"):
                path = root / f"{name}.engine"
                path.write_bytes(name.encode())
                engines[name] = path
                artifacts[f"tensorrt_{name}"] = {"sha256": BENCHMARK.sha256(path)}
            evaluation = root / "evaluation.json"
            evaluation.write_text(json.dumps({
                "schema_version": 1,
                "artifacts": artifacts,
                "backends": {"tensorrt_int8": {"passed": False}},
            }), encoding="utf-8")
            _, eligible = BENCHMARK.load_evaluation(evaluation, engines)
            self.assertFalse(eligible)

    def test_trtexec_version_parser_supports_tensorrt_10(self) -> None:
        original = BENCHMARK.command_output
        BENCHMARK.command_output = lambda command: "RUNNING [TensorRT v101401]"
        try:
            self.assertEqual(BENCHMARK.trtexec_version("trtexec"), "10.14.1")
        finally:
            BENCHMARK.command_output = original


if __name__ == "__main__":
    unittest.main()
