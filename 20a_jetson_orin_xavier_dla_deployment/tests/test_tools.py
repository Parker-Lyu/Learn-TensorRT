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


BENCHMARK = load("benchmark_target.py")
FALLBACK = load("analyze_fallback.py")
PLATFORM = load("check_platform.py")


class JetsonLessonTests(unittest.TestCase):
    def test_benchmark_requires_100_samples(self):
        with self.assertRaisesRegex(ValueError, "100"):
            BENCHMARK.summarize([{"latencyMs": 1, "computeMs": 1}] * 99)

    def test_fallback_parser(self):
        result = FALLBACK.analyze("DeviceType: DLA\nlayer cannot run on DLA, fallback GPU")
        self.assertGreaterEqual(result["dla_assignment_mentions"], 1)
        self.assertEqual(len(result["fallback_warnings"]), 1)

    def test_platform_manifest_has_compatibility_fields(self):
        result = PLATFORM.detect()
        for field in ("machine", "is_jetson", "tensorrt", "dla_cores", "kernel"):
            self.assertIn(field, result)


if __name__ == "__main__":
    unittest.main()
