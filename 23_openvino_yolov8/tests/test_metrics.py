import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "run_openvino.py"
SPEC = importlib.util.spec_from_file_location("run_openvino", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)

COMPARISON_PATH = Path(__file__).resolve().parents[1] / "generate_comparison.py"
COMPARISON_SPEC = importlib.util.spec_from_file_location("generate_comparison", COMPARISON_PATH)
COMPARISON = importlib.util.module_from_spec(COMPARISON_SPEC)
assert COMPARISON_SPEC.loader
COMPARISON_SPEC.loader.exec_module(COMPARISON)


def comparison_fixtures(include_int8=True):
    latency = {"mean": 2.0, "p50": 1.9, "p90": 2.2, "p99": 2.5}
    ov = {
        "hardware": {"cpu_model": "test-cpu", "logical_cpu_count": 8},
        "software": {"openvino": "2025.4.1"},
        "sync": {"latency_ms": latency, "throughput_requests_per_second": 100.0},
        "async": {"latency_ms": latency, "throughput_requests_per_second": 200.0},
        "alignment_vs_onnxruntime": {"max_abs": 0.1, "mean_abs": 0.01, "p99_abs": 0.05},
    }
    trt = {
        "schema_version": 3,
        "environment": {"gpu": "test-gpu", "trtexec": "10.14.1"},
        "backends": {
            precision: {"latency_ms": latency, "throughput_qps": 1000.0}
            for precision in (("fp32", "fp16", "int8") if include_int8 else ("fp32", "fp16"))
        },
    }
    return ov, trt


class MetricTests(unittest.TestCase):
    def test_nearest_rank_percentile(self):
        self.assertEqual(MODULE.percentile(list(range(1, 101)), 0.99), 99)

    def test_summary_throughput(self):
        result = MODULE.summary([2.0] * 100, 0.5)
        self.assertEqual(result["throughput_requests_per_second"], 200.0)


class ComparisonTests(unittest.TestCase):
    def test_renders_quality_gated_int8_when_present(self):
        report = COMPARISON.render_comparison(*comparison_fixtures())
        self.assertIn("TensorRT GPU INT8", report)
        self.assertNotIn("INT8 is omitted", report)

    def test_renders_report_when_failed_int8_has_no_performance(self):
        report = COMPARISON.render_comparison(*comparison_fixtures(include_int8=False))
        self.assertNotIn("TensorRT GPU INT8", report)
        self.assertIn("INT8 is omitted", report)

    def test_rejects_missing_required_tensorrt_backend(self):
        ov, trt = comparison_fixtures()
        del trt["backends"]["fp16"]
        with self.assertRaisesRegex(ValueError, "missing required backends: fp16"):
            COMPARISON.render_comparison(ov, trt)


if __name__ == "__main__":
    unittest.main()
