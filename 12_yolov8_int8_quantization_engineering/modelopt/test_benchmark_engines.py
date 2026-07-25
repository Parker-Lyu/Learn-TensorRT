#!/usr/bin/env python3
"""Focused tests for matched TensorRT 10 performance reporting."""

from __future__ import annotations

import importlib.util
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


if __name__ == "__main__":
    unittest.main()

