import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "tools/analyze_calibration_representativeness.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_calibration_representativeness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RepresentativenessTests(unittest.TestCase):
    def test_ks_distance_is_zero_for_identical_samples(self) -> None:
        values = np.asarray([1.0, 1.0, 2.0, 4.0])
        self.assertAlmostEqual(MODULE.ks_distance(values, np.sort(values)), 0.0)

    def test_ks_distance_detects_separated_samples(self) -> None:
        sample = np.asarray([10.0, 11.0, 12.0])
        population = np.asarray([0.0, 1.0, 2.0])
        self.assertAlmostEqual(MODULE.ks_distance(sample, population), 1.0)

    def test_geometry_excludes_crowd_and_builds_relative_boxes(self) -> None:
        document = {
            "images": [
                {"id": 2, "width": 200, "height": 100},
                {"id": 1, "width": 100, "height": 100},
            ],
            "annotations": [
                {"image_id": 1, "bbox": [0, 0, 10, 20], "area": 200, "iscrowd": 0},
                {"image_id": 2, "bbox": [0, 0, 50, 20], "area": 1000, "iscrowd": 0},
                {"image_id": 2, "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 1},
            ],
        }
        image_ids, image_metrics, box_metrics, metadata = MODULE.build_geometry(document)
        self.assertEqual(image_ids, [1, 2])
        np.testing.assert_allclose(image_metrics["objects_per_image"], [1.0, 1.0])
        np.testing.assert_allclose(box_metrics["box_relative_area"][0], [0.02])
        np.testing.assert_allclose(box_metrics["box_relative_area"][1], [0.05])
        self.assertEqual(metadata["ignored_crowd_annotations"], 1)

    def test_support_coverage_reports_missing_population_interval(self) -> None:
        population = np.arange(100, dtype=np.float64)
        sample = np.arange(10, dtype=np.float64)
        result = MODULE.support_coverage(sample, population)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["empty_bins"])


if __name__ == "__main__":
    unittest.main()
