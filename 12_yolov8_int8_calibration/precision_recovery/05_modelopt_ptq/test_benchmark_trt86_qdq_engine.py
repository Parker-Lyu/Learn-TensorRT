#!/usr/bin/env python3
"""Tests for matched ModelOpt Q/DQ throughput reporting."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "benchmark_trt86_qdq_engine.py"
SPEC = importlib.util.spec_from_file_location("benchmark_trt86_qdq_engine", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkTrt86QdqEngineTests(unittest.TestCase):
    def test_parse_throughput_uses_final_summary(self) -> None:
        text = "Throughput: 1.0 qps\nThroughput: 812.5 qps"
        self.assertEqual(BENCHMARK.parse_throughput(text), 812.5)

    def test_summary_uses_measured_samples(self) -> None:
        samples = [
            {"latencyMs": index + 1, "computeMs": 1, "h2dMs": 2, "d2hMs": 3}
            for index in range(100)
        ]
        summary = BENCHMARK.summarize(samples)
        self.assertEqual(summary["sample_count"], 100)
        self.assertEqual(summary["latency_ms"]["p90"], 90.0)
        self.assertEqual(summary["gpu_compute_ms"]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
