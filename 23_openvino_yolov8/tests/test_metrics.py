import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "run_openvino.py"
SPEC = importlib.util.spec_from_file_location("run_openvino", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class MetricTests(unittest.TestCase):
    def test_nearest_rank_percentile(self):
        self.assertEqual(MODULE.percentile(list(range(1, 101)), 0.99), 99)

    def test_summary_throughput(self):
        result = MODULE.summary([2.0] * 100, 0.5)
        self.assertEqual(result["throughput_requests_per_second"], 200.0)


if __name__ == "__main__":
    unittest.main()
