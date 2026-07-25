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

    def test_summary_computes_latency_without_inventing_throughput(self):
        summary = MODULE.summarize([{"latencyMs": 4, "computeMs": 3}] * 100)
        self.assertEqual(summary["latency_ms"]["p99"], 4.0)
        self.assertNotIn("throughput_qps", summary)

    def test_throughput_uses_trtexec_wall_time_result(self):
        output = "[I] Throughput: 800.602 qps\n[I] Latency: mean = 2.37974 ms"
        self.assertEqual(MODULE.parse_trtexec_throughput(output), 800.602)

    def test_missing_throughput_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "throughput"):
            MODULE.parse_trtexec_throughput("[I] Latency: mean = 2.0 ms")

    def test_non_positive_throughput_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "invalid throughput"):
            MODULE.parse_trtexec_throughput("[I] Throughput: 0 qps")

    def test_trtexec_version_parser_supports_tensorrt_10(self):
        original = MODULE.command_output
        MODULE.command_output = lambda command: "RUNNING [TensorRT v101401]"
        try:
            self.assertEqual(MODULE.trtexec_version("trtexec"), "10.14.1")
        finally:
            MODULE.command_output = original


if __name__ == "__main__":
    unittest.main()
