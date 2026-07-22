import importlib.util
import hashlib
import tempfile
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
    def test_committed_selection_contract_is_well_formed(self) -> None:
        contract = MODULE.load_selection_contract(MODULE.DEFAULT_SELECTION_CONTRACT)
        self.assertEqual(len(contract["candidate_pool"]), 5000)
        self.assertEqual(len(contract["selected_ids"]), 3000)

    def test_candidate_download_is_verified_before_reuse(self) -> None:
        payload = b"fixed-coco-image"
        record = {
            "image_id": 42,
            "file_name": "000000000042.jpg",
            "image_sha256": hashlib.sha256(payload).hexdigest(),
        }
        calls = []

        def downloader(url: str, destination: Path) -> None:
            calls.append(url)
            destination.write_bytes(payload)

        with tempfile.TemporaryDirectory() as directory:
            stale = Path(directory) / record["file_name"]
            stale.write_bytes(b"stale-content")
            destination = MODULE.ensure_candidate_image(
                record, Path(directory), downloader=downloader
            )
            self.assertEqual(destination.read_bytes(), payload)
            MODULE.ensure_candidate_image(record, Path(directory), downloader=downloader)
        self.assertEqual(len(calls), 1)

    def test_selection_mismatch_does_not_modify_contract(self) -> None:
        contract = {"selected_ids": [10, 20]}
        regenerated = {"selected_ids": [10, 30]}
        with self.assertRaisesRegex(RuntimeError, "expected image 20"):
            MODULE.require_selection_match(contract, regenerated)
        self.assertEqual(contract["selected_ids"], [10, 20])

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
