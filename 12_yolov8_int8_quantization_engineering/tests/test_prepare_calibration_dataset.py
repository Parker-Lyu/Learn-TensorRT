import importlib.util
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1] / "tools/prepare_calibration_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_calibration_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def profile(image_id: int, classes: tuple[int, ...]) -> dict:
    return {
        "id": image_id,
        "classes": classes,
        "class_counts": Counter(classes),
        "size_counts": Counter({"small": 1}),
        "object_count": len(classes),
    }


class CalibrationSelectionTests(unittest.TestCase):
    def test_category_seeding_covers_every_class(self) -> None:
        profiles = [profile(index, (index,)) for index in range(80)]
        selected = MODULE.category_seeds(profiles)
        covered = {class_id for index in selected for class_id in profiles[index]["classes"]}
        self.assertEqual(covered, set(range(80)))

    def test_selection_is_deterministic_and_unique(self) -> None:
        profiles = [profile(index, (index % 80,)) for index in range(160)]
        features = np.arange(160 * 12, dtype=np.float64).reshape(160, 12)
        first = MODULE.coverage_selection(profiles, features, 100)
        second = MODULE.coverage_selection(profiles, features, 100)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))

    def test_rejects_target_larger_than_pool(self) -> None:
        profiles = [profile(index, (index,)) for index in range(80)]
        features = np.zeros((80, 12), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "target count"):
            MODULE.coverage_selection(profiles, features, 81)


if __name__ == "__main__":
    unittest.main()
