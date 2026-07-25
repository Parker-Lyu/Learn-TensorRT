import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_report.py"
SPEC = importlib.util.spec_from_file_location("generate_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def fixtures():
    manifest_hash = "a" * 64
    manifest = {
        "dataset_id": "fixed-v1",
        "calibration_count": 1,
        "validation_count": 100,
        "records": [
            {"split": "calibration", "image_sha256": "c" * 64},
            *[
                {"split": "validation", "image_sha256": f"{index:064x}"}
                for index in range(100)
            ],
        ],
    }
    performance = {
        "schema_version": 3,
        "methodology": {
            "warmup_ms": 500,
            "iterations": 120,
            "synchronization": "synchronized",
        },
        "environment": {"gpu": "test-gpu", "trtexec": "10.14.1"},
        "backends": {},
    }
    for index, key in enumerate(("fp32", "fp16", "int8"), start=1):
        performance["backends"][key] = {
            "sample_count": 120,
            "engine_sha256": str(index) * 64,
            "latency_ms": {"mean": 5.0 - index, "p50": 1.0, "p90": 2.0, "p99": 3.0},
            "throughput_qps": 250.0,
        }
    metrics = {"map50_95": 0.4, "map50": 0.6, "precision": 0.7, "recall": 0.8}
    evaluation = {
        "schema_version": 1,
        "dataset": {
            "dataset_id": "fixed-v1",
            "validation_images": 100,
            "manifest_sha256": manifest_hash,
        },
        "settings": {
            "confidence": 0.001,
            "nms_iou": 0.7,
            "max_detections": 300,
            "metric_implementation": "course-v2",
            "latency_scope": "matched transfers",
        },
        "software": {"tensorrt": "10.14.1"},
        "regression_thresholds": {
            "max_map50_95_drop": 0.02,
            "max_map50_drop": 0.02,
            "max_precision_drop": 0.03,
            "max_recall_drop": 0.03,
        },
        "artifacts": {},
        "backends": {
            "pytorch": {"passed": True, "metrics": metrics, "delta_vs_pytorch": {}},
        },
    }
    for index, key in enumerate(("tensorrt_fp32", "tensorrt_fp16", "tensorrt_int8"), start=1):
        passed = key != "tensorrt_int8"
        evaluation["artifacts"][key] = {"sha256": str(index) * 64}
        evaluation["backends"][key] = {
            "passed": passed,
            "metrics": metrics,
            "delta_vs_pytorch": {"map50_95": 0.0},
            "tensor_drift_vs_fp32": {"max_abs": 0.1, "mean_abs": 0.01, "p99_abs": 0.02},
        }
    evaluation["release_gate"] = {
        "passed": False,
        "failed_backends": ["tensorrt_int8"],
    }
    diagnosis = {
        "schema_version": 2,
        "artifacts": {"engine": {"sha256": "1" * 64}},
        "baseline_summary": {"heuristic_diagnosis": {"diagnosis": "CPU work dominates."}}
    }
    return performance, evaluation, diagnosis, manifest, manifest_hash


class GenerateReportTests(unittest.TestCase):
    def test_report_renders_dynamic_decisions_and_drift(self):
        text = MODULE.render(*fixtures())
        self.assertIn("INT8 is faster than FP32 and fails", text)
        self.assertIn("Raw Tensor Drift", text)
        self.assertIn("course-v2", text)

    def test_passing_but_slower_int8_retains_fp16(self):
        performance, evaluation, diagnosis, manifest, manifest_hash = fixtures()
        evaluation["backends"]["tensorrt_int8"]["passed"] = True
        evaluation["release_gate"] = {"passed": True, "failed_backends": []}
        performance["backends"]["fp16"]["throughput_qps"] = 600.0
        performance["backends"]["int8"]["throughput_qps"] = 500.0
        performance["backends"]["fp16"]["latency_ms"]["mean"] = 2.0
        performance["backends"]["int8"]["latency_ms"]["mean"] = 3.0
        text = MODULE.render(performance, evaluation, diagnosis, manifest, manifest_hash)
        self.assertIn("retain FP16 for deployment", text)

    def test_engine_identity_mismatch_is_rejected(self):
        evidence = list(fixtures())
        evidence[0]["backends"]["int8"]["engine_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "int8 engine differs"):
            MODULE.render(*evidence)

    def test_dataset_identity_mismatch_is_rejected(self):
        evidence = list(fixtures())
        evidence[1]["dataset"]["dataset_id"] = "other"
        with self.assertRaisesRegex(ValueError, "dataset_id"):
            MODULE.render(*evidence)

    def test_short_performance_sample_is_rejected(self):
        evidence = list(fixtures())
        evidence[0]["backends"]["fp16"]["sample_count"] = 99
        with self.assertRaisesRegex(ValueError, "fewer than 100"):
            MODULE.render(*evidence)

    def test_invalid_throughput_is_rejected(self):
        evidence = list(fixtures())
        evidence[0]["backends"]["fp16"]["throughput_qps"] = 0.0
        with self.assertRaisesRegex(ValueError, "invalid trtexec throughput"):
            MODULE.render(*evidence)

    def test_profiler_engine_is_contextual_without_matched_identity(self):
        evidence = list(fixtures())
        evidence[2]["artifacts"]["engine"]["sha256"] = "f" * 64
        text = MODULE.render(*evidence)
        self.assertIn("Lesson 11 Nsight baseline", text)


if __name__ == "__main__":
    unittest.main()
