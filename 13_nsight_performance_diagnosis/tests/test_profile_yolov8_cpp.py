#!/usr/bin/env python3

import importlib.util
import math
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "profile_yolov8_cpp.py"
SPEC = importlib.util.spec_from_file_location("profile_yolov8_cpp", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PROFILE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROFILE
SPEC.loader.exec_module(PROFILE)


def sample(**overrides: float) -> dict[str, float]:
    values = {
        "preprocess": 4.0,
        "h2d": 1.0,
        "enqueue_host": 0.2,
        "gpu_compute": 3.0,
        "d2h": 1.0,
        "postprocess": 1.0,
        "total": 10.0,
    }
    values.update(overrides)
    return values


class PercentileTests(unittest.TestCase):
    def test_nearest_rank_percentiles(self) -> None:
        values = [4.0, 1.0, 3.0, 2.0]
        self.assertEqual(PROFILE.percentile(values, 50), 2.0)
        self.assertEqual(PROFILE.percentile(values, 99), 4.0)

    def test_rejects_invalid_input(self) -> None:
        with self.assertRaises(ValueError):
            PROFILE.percentile([], 50)
        with self.assertRaises(ValueError):
            PROFILE.percentile([1.0], 101)


class ValidationTests(unittest.TestCase):
    def test_accepts_complete_non_negative_samples(self) -> None:
        self.assertEqual(PROFILE.validate_latency_samples([sample()], "samples"), [sample()])

    def test_rejects_missing_and_non_finite_values(self) -> None:
        missing = sample()
        del missing["gpu_compute"]
        with self.assertRaisesRegex(ValueError, "gpu_compute"):
            PROFILE.validate_latency_samples([missing], "samples")
        with self.assertRaisesRegex(ValueError, "total"):
            PROFILE.validate_latency_samples([sample(total=math.inf)], "samples")


class DiagnosisTests(unittest.TestCase):
    def test_composition_is_calculated_per_request(self) -> None:
        composition = PROFILE.summarize_composition([sample(), sample(preprocess=5.0, total=11.0)])
        self.assertAlmostEqual(composition["gpu_compute"]["p50"], 3.0 / 11.0)
        self.assertAlmostEqual(composition["cpu"]["p50"], 0.5)

    def test_requires_a_clear_lead_before_declaring_dominance(self) -> None:
        mixed = PROFILE.summarize_composition([sample(preprocess=3.0, gpu_compute=4.0)])
        self.assertIsNone(PROFILE.classify_bottleneck(mixed)["dominant_category"])

        gpu_heavy = PROFILE.summarize_composition(
            [sample(preprocess=1.0, h2d=0.5, gpu_compute=7.0, d2h=0.5, postprocess=0.5)]
        )
        self.assertEqual(PROFILE.classify_bottleneck(gpu_heavy)["dominant_category"], "gpu_compute")


if __name__ == "__main__":
    unittest.main()
