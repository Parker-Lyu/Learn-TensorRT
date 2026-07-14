import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "collect_performance.py"
SPEC = importlib.util.spec_from_file_location("collect_performance", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class PerformanceSummaryTests(unittest.TestCase):
    def test_percentile_uses_nearest_rank(self):
        self.assertEqual(MODULE.percentile(list(range(1, 101)), 0.99), 99)

    def test_summary_requires_enough_samples_for_p99(self):
        with self.assertRaisesRegex(ValueError, "100"):
            MODULE.summarize([{"latencyMs": 1, "computeMs": 1}] * 99)

    def test_summary_computes_throughput(self):
        summary = MODULE.summarize([{"latencyMs": 4, "computeMs": 3}] * 100)
        self.assertEqual(summary["throughput_images_per_second"], 250.0)
        self.assertEqual(summary["latency_ms"]["p99"], 4.0)

    def test_trtexec_version_parser_uses_banner(self):
        original = MODULE.command_output
        MODULE.command_output = lambda command: "RUNNING [TensorRT v8601]\nerror after banner"
        try:
            self.assertEqual(MODULE.trtexec_version("trtexec"), "8.6.1")
        finally:
            MODULE.command_output = original


if __name__ == "__main__":
    unittest.main()
