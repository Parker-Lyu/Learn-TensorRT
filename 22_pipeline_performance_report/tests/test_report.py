import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COLLECT = load("collect_pipeline_evidence.py")
REPORT = load("generate_report.py")


def passing_evidence():
    metrics = {"captured": 10, "completed": 10, "evicted": 0,
               "failed": 0, "aborted": 0}
    run = {"returncode": 0, "bounded": True, "metrics": metrics}
    return {
        "schema_version": 3,
        "platform": {key: "recorded" for key in
                     ("development_image", "gpu", "compute_capability", "driver", "tensorrt",
                      "cuda_runtime", "cuda_driver", "collected_at")},
        "load_matrix": {
            "batches": {str(size): {"returncode": 0,
                         "metrics": {"batch_distribution": {str(size): 1}}}
                        for size in (1, 2, 4)},
            "two_slot_overlap": {"returncode": 0, "overlap_observed": True},
        },
        "reference_checks": {
            name: {"returncode": 0, "within_tolerance": True}
            for name in ("batch_vs_single", "cpu_vs_cuda_preprocess")},
        "multi_stream": {"returncode": 0, "producer": "lesson21", "metrics": {"streams": [
            {"stream_id": 0, "p50_ms": 1.0, "fps": 10.0},
            {"stream_id": 1, "p50_ms": 1.1, "fps": 9.0}]}},
        "policies": {name: dict(run) for name in ("block", "drop_oldest", "latest_first")},
        "faults": {name: {"returncode": 1, "cleanup_complete": True} for name in
                   ("source_read", "invalid_shape", "insufficient_capacity", "tensor_address",
                    "enqueue", "postprocess", "abort_pending")},
        "restarts": {"requested": 100, "failures": 0},
        "long_lived_soak": {"returncode": 0, "single_process": True,
                            "actual_seconds": 1800.0, "failures": 0},
        "memory_trend": {name: {"sample_count": 10, "growth_percent": 1.0,
                                "threshold_percent": 5.0} for name in ("host", "device")},
        "sanitizers": {name: {"returncode": 0, "tool_started": True} for name in
                       ("compute_memcheck_lesson21", "lesson21_cpu_tsan")},
    }


class PipelineReportTests(unittest.TestCase):
    def test_single_parser(self):
        parsed = COLLECT.parse_single(
            "captured=10 processed=7 dropped=3 queue_peak=2 fps=10 "
            "p50_ms=1 p90_ms=2 p99_ms=3")
        self.assertEqual(parsed["processed"], 7)

    def test_complete_fixture_passes(self):
        gates = REPORT.evaluate(passing_evidence())
        self.assertEqual(REPORT.overall_status(gates), REPORT.PASS)
        self.assertTrue(all(gate["status"] == REPORT.PASS for gate in gates.values()))

    def test_missing_fixture_is_incomplete_without_exception(self):
        gates = REPORT.evaluate({})
        self.assertEqual(REPORT.overall_status(gates), REPORT.INCOMPLETE)
        self.assertTrue(all(gate["status"] == REPORT.INCOMPLETE for gate in gates.values()))

    def test_malformed_fixture_is_incomplete_without_exception(self):
        evidence = passing_evidence()
        evidence["restarts"] = {"requested": "not-a-number", "failures": 0}
        gates = REPORT.evaluate(evidence)
        self.assertEqual(gates["restart_100"]["status"], REPORT.INCOMPLETE)
        self.assertEqual(REPORT.overall_status(gates), REPORT.INCOMPLETE)

    def test_executed_failure_fails_report(self):
        evidence = passing_evidence()
        evidence["long_lived_soak"]["failures"] = 1
        gates = REPORT.evaluate(evidence)
        self.assertEqual(gates["soak_30_minutes"]["status"], REPORT.FAIL)
        self.assertEqual(REPORT.overall_status(gates), REPORT.FAIL)

    def test_short_soak_fails_when_it_was_executed(self):
        evidence = passing_evidence()
        evidence["long_lived_soak"]["actual_seconds"] = 60.0
        gates = REPORT.evaluate(evidence)
        self.assertEqual(gates["soak_30_minutes"]["status"], REPORT.FAIL)


if __name__ == "__main__":
    unittest.main()
