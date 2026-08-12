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
        "schema_version": 2,
        "hardware": {"cpu_model": "test-cpu", "logical_cpu_count": 8},
        "software": {"openvino": "2025.4.1"},
        "sync": {"latency_ms": latency, "throughput_requests_per_second": 100.0},
        "async": {
            "response_latency_ms": latency,
            "inference_latency_ms": latency,
            "throughput_requests_per_second": 200.0,
            "jobs_requested": 0,
            "jobs_actual": 8,
            "optimal_number_of_infer_requests": 8,
        },
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

    def test_latency_summary_rejects_empty_measurements(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            MODULE.latency_summary([])


class ComparisonTests(unittest.TestCase):
    def test_renders_quality_gated_int8_when_present(self):
        report = COMPARISON.render_comparison(*comparison_fixtures())
        self.assertIn("TensorRT GPU INT8", report)
        self.assertNotIn("INT8 is omitted", report)
        self.assertIn("submit-to-completion response", report)
        self.assertIn("InferRequest execution", report)
        self.assertIn("selected 8 async requests", report)

    def test_renders_report_when_failed_int8_has_no_performance(self):
        report = COMPARISON.render_comparison(*comparison_fixtures(include_int8=False))
        self.assertNotIn("TensorRT GPU INT8", report)
        self.assertIn("INT8 is omitted", report)

    def test_rejects_missing_required_tensorrt_backend(self):
        ov, trt = comparison_fixtures()
        del trt["backends"]["fp16"]
        with self.assertRaisesRegex(ValueError, "missing required backends: fp16"):
            COMPARISON.render_comparison(ov, trt)

    def test_legacy_openvino_evidence_is_labeled(self):
        ov, trt = comparison_fixtures()
        del ov["schema_version"]
        ov["async"]["latency_ms"] = ov["async"].pop("response_latency_ms")
        del ov["async"]["inference_latency_ms"]
        report = COMPARISON.render_comparison(ov, trt)
        self.assertIn("Legacy OpenVINO evidence", report)
        self.assertNotIn("| OpenVINO CPU async | InferRequest execution |", report)


if __name__ == "__main__":
    unittest.main()
