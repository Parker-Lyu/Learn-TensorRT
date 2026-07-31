import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_report.py"
SPEC = importlib.util.spec_from_file_location("generate_report", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def fixtures(int8_passed=False):
    manifest_hash = "a" * 64
    evaluation_hash = "e" * 64
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
            "inference_streams": 1,
            "data_transfers": True,
        },
        "environment": {"gpu": "test-gpu", "trtexec": "10.14.1"},
        "quality_gate": {
            "evaluation_sha256": evaluation_hash,
            "int8_eligible_for_performance": int8_passed,
        },
        "backends": {},
    }
    performance_keys = ("fp32", "fp16", "int8") if int8_passed else ("fp32", "fp16")
    for index, key in enumerate(performance_keys, start=1):
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
        passed = key != "tensorrt_int8" or int8_passed
        evaluation["artifacts"][key] = {"sha256": str(index) * 64}
        evaluation["backends"][key] = {
            "passed": passed,
            "metrics": metrics,
            "delta_vs_pytorch": {"map50_95": 0.0},
            "tensor_drift_vs_fp32": {"max_abs": 0.1, "mean_abs": 0.01, "p99_abs": 0.02},
        }
    evaluation["release_gate"] = {
        "passed": int8_passed,
        "failed_backends": [] if int8_passed else ["tensorrt_int8"],
    }
    diagnosis = {
        "schema_version": 1,
        "engines": {
            "tensorrt_int8": {
                "compute_output_precision_counts": {"INT8": 44, "FP16": 50, "FP32": 2}
            }
        },
    }
    return performance, evaluation, diagnosis, manifest, manifest_hash, evaluation_hash


class GenerateReportTests(unittest.TestCase):
    def test_report_renders_dynamic_decisions_and_drift(self):
        text = MODULE.render(*fixtures())
        self.assertIn("INT8 fails the accuracy gate and was not benchmarked", text)
        self.assertIn("Raw Tensor Drift", text)
        self.assertIn("course-v2", text)

    def test_passing_but_slower_int8_retains_fp16(self):
        performance, evaluation, diagnosis, manifest, manifest_hash, evaluation_hash = fixtures(True)
        performance["backends"]["fp16"]["throughput_qps"] = 600.0
        performance["backends"]["int8"]["throughput_qps"] = 500.0
        performance["backends"]["fp16"]["latency_ms"]["mean"] = 2.0
        performance["backends"]["int8"]["latency_ms"]["mean"] = 3.0
        text = MODULE.render(
            performance, evaluation, diagnosis, manifest, manifest_hash, evaluation_hash
        )
        self.assertIn("retain FP16 for deployment", text)

    def test_engine_identity_mismatch_is_rejected(self):
        evidence = list(fixtures(True))
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

    def test_failed_int8_performance_is_rejected(self):
        evidence = list(fixtures())
        evidence[0]["backends"]["int8"] = dict(evidence[0]["backends"]["fp16"])
        with self.assertRaisesRegex(ValueError, "contain INT8 only after"):
            MODULE.render(*evidence)

    def test_performance_must_reference_the_same_evaluation(self):
        evidence = list(fixtures())
        evidence[0]["quality_gate"]["evaluation_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "different precision evaluation"):
            MODULE.render(*evidence)

    def test_layer_audit_context_is_rendered(self):
        evidence = list(fixtures())
        text = MODULE.render(*evidence)
        self.assertIn("TensorRT 10.14 layer audit", text)


if __name__ == "__main__":
    unittest.main()
