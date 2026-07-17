import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parent / "prepare_coverage_calibration.py"
SPEC = importlib.util.spec_from_file_location("prepare_coverage_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
COVERAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COVERAGE)


def synthetic_document() -> dict:
    categories = [{"id": index + 1} for index in range(80)]
    images = [
        {"id": index + 1, "file_name": f"{index + 1:012d}.jpg", "width": 640, "height": 480}
        for index in range(100)
    ]
    annotations = []
    for index, image in enumerate(images):
        annotations.append({
            "image_id": image["id"],
            "category_id": index % 80 + 1,
            "area": [400.0, 2500.0, 20_000.0][index % 3],
            "iscrowd": 0,
        })
    return {"categories": categories, "images": images, "annotations": annotations}


class CalibrationCoverageTests(unittest.TestCase):
    def test_profiles_map_categories_and_object_sizes(self) -> None:
        profiles = COVERAGE.build_profiles(synthetic_document())
        self.assertEqual(profiles[1]["classes"], (0,))
        self.assertEqual(profiles[1]["size_counts"]["small"], 1)
        self.assertEqual(profiles[2]["size_counts"]["medium"], 1)
        self.assertEqual(profiles[3]["size_counts"]["large"], 1)

    def test_annotation_pool_is_deterministic_and_excludes_baseline(self) -> None:
        profiles = COVERAGE.build_profiles(synthetic_document())
        first = COVERAGE.annotation_pool(profiles, {1, 2, 3}, 50, 42)
        second = COVERAGE.annotation_pool(profiles, {1, 2, 3}, 50, 42)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 50)
        self.assertFalse(set(first) & {1, 2, 3})

    def test_farthest_selection_prefers_uncovered_extremes(self) -> None:
        base = np.asarray([[0.0], [0.1]], dtype=np.float64)
        candidates = np.asarray([[0.2], [0.5], [1.0]], dtype=np.float64)
        selected = COVERAGE.farthest_coverage_selection(base, candidates, 2)
        self.assertEqual(selected[0], 2)
        self.assertIn(1, selected)

    def test_numeric_features_are_finite(self) -> None:
        document = synthetic_document()
        profiles = COVERAGE.build_profiles(document)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            image = np.full((32, 64, 3), (10, 100, 220), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), image))
            features = COVERAGE.numeric_features(profiles[1], path)
        self.assertEqual(features.shape, (12,))
        self.assertTrue(np.all(np.isfinite(features)))


if __name__ == "__main__":
    unittest.main()
