import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools/generate_run_summary.py"
SPEC = importlib.util.spec_from_file_location("generate_run_summary", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def fixtures(int8_passed: bool):
    parity = {"status": "PASS"}
    representative = {"conclusion": {"calibration_selection_status": "PASS"}}
    qdq = {
        "calibration_images": 3000,
        "onnx_sha256": "a" * 64,
        "onnx_inspection": {
            "checker_passed": True,
            "quantize_linear_nodes": 10,
            "dequantize_linear_nodes": 10,
        },
    }
    metrics = {"map50_95": 0.4, "map50": 0.6, "precision": 0.7, "recall": 0.8}
    evaluation = {
        "dataset": {"dataset_id": "fixed", "validation_images": 5000},
        "backends": {"tensorrt_int8": {"passed": int8_passed, "metrics": metrics}},
        "release_gate": {
            "passed": int8_passed,
            "failed_backends": [] if int8_passed else ["tensorrt_int8"],
        },
    }
    audit = {"engines": {"tensorrt_int8": {"compute_output_precision_counts": {"INT8": 40}}}}
    sensitivity = {"passed": True}
    backend = {
        "sample_count": 120,
        "latency_ms": {"mean": 2.0, "p99": 3.0},
        "throughput_qps": 500.0,
    }
    performance = {
        "schema_version": 3,
        "backends": {"fp32": backend, "fp16": backend},
    }
    if int8_passed:
        performance["backends"]["int8"] = backend
    return parity, representative, qdq, evaluation, audit, sensitivity, performance


class RunSummaryTests(unittest.TestCase):
    def test_failed_int8_is_not_benchmarked(self):
        text = MODULE.render(*fixtures(False))
        self.assertIn("REJECTED", text)
        self.assertIn("INT8 was not benchmarked", text)

    def test_passing_int8_is_in_performance_table(self):
        text = MODULE.render(*fixtures(True))
        self.assertIn("ACCEPTED FOR PERFORMANCE COMPARISON", text)
        self.assertIn("| INT8 | 120 |", text)


if __name__ == "__main__":
    unittest.main()
