import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


METRICS = load("metrics.py")
PREPARE = load("prepare_model_repository.py")


class TritonLessonTests(unittest.TestCase):
    def test_percentiles_and_throughput(self):
        result = METRICS.summarize(list(range(1, 101)), 0.5)
        self.assertEqual(result["latency_ms"]["p99"], 99)
        self.assertEqual(result["throughput_requests_per_second"], 200)

    def test_committed_config_has_dynamic_batching(self):
        text = (ROOT / "model_repository/yolov8/config.pbtxt").read_text()
        PREPARE.validate_config(text)


if __name__ == "__main__":
    unittest.main()
