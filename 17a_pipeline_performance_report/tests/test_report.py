import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


COLLECT = load("collect_pipeline_evidence.py")
REPORT = load("generate_report.py")


class PipelineReportTests(unittest.TestCase):
    def test_single_parser(self):
        parsed = COLLECT.parse_single(
            "captured=10 processed=7 dropped=3 queue_peak=2 fps=10 p50_ms=1 p90_ms=2 p99_ms=3")
        self.assertEqual(parsed["processed"], 7)

    def test_gate_keeps_short_soak_incomplete(self):
        evidence = {
            "single_stream": {"metrics": {"queue_peak": 4, "captured": 10,
                                             "processed": 7, "dropped": 3}},
            "multi_stream": {"metrics": {"streams": [{"queue_peak": 4}, {"queue_peak": 3}]}},
            "restarts": {"requested": 100, "failures": 0},
            "soak": {"requested_minutes": 1, "failures": 0},
            "faults": {"x": {"expected_nonzero": True}},
            "sanitizers": {"compute_memcheck": {"returncode": 0},
                           "thread_sanitizer": {"returncode": 0}},
        }
        self.assertFalse(REPORT.evaluate(evidence)["soak_30_minutes"])


if __name__ == "__main__":
    unittest.main()
