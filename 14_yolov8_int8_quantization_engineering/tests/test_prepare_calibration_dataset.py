import importlib.util
import hashlib
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools/prepare_calibration_dataset.py"
SPEC = importlib.util.spec_from_file_location("prepare_calibration_dataset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def profile(image_id: int, classes: tuple[int, ...], objects: int = 1) -> dict:
    return {
        "image_id": image_id,
        "file_name": f"{image_id:012d}.jpg",
        "width": 640,
        "height": 480,
        "classes": classes,
        "object_count": objects,
        "min_relative_box_area": 0.001 * image_id,
        "max_relative_box_area": 0.01 * image_id,
        "aspect_score": 0.1 * image_id,
    }


class CalibrationSelectionTests(unittest.TestCase):
    def test_committed_config_declares_eighty_twenty_split(self) -> None:
        config = MODULE.validate_config(MODULE.load_json(MODULE.DEFAULT_CONFIG))
        self.assertEqual(config["counts"]["natural_core"], 2400)
        self.assertEqual(config["counts"]["tail_total"], 600)
        self.assertEqual(config["counts"]["per_tail_group"], 100)

    def test_committed_contract_matches_config_and_role_counts(self) -> None:
        contract = MODULE.load_json(MODULE.DEFAULT_CONTRACT)
        config_hash = hashlib.sha256(MODULE.DEFAULT_CONFIG.read_bytes()).hexdigest()
        self.assertEqual(contract["config_sha256"], config_hash)
        self.assertEqual(len(contract["brightness_screen"]), 3000)
        roles = MODULE.Counter(record["role"] for record in contract["selected_images"])
        self.assertEqual(roles["natural_core"], 2400)
        for group in MODULE.load_json(MODULE.DEFAULT_CONFIG)["tail_groups"]:
            self.assertEqual(roles[group], 100)
        self.assertEqual(sum(roles.values()), 3000)

    def test_natural_core_is_deterministic_and_repairs_category_coverage(self) -> None:
        profiles = {index: profile(index, (index % 80,)) for index in range(1, 401)}
        first, first_swaps = MODULE.natural_core(profiles, 200, 42)
        second, second_swaps = MODULE.natural_core(profiles, 200, 42)
        self.assertEqual(first, second)
        self.assertEqual(first_swaps, second_swaps)
        covered = {class_id for image_id in first for class_id in profiles[image_id]["classes"]}
        self.assertEqual(covered, set(range(80)))

    def test_geometry_tail_is_unique_from_excluded_core(self) -> None:
        profiles = {index: profile(index, (index % 80,)) for index in range(1, 1001)}
        selected, _ = MODULE.choose_geometry_tail(
            profiles, set(range(1, 101)), "crowded", 25, 0.9, 7
        )
        self.assertEqual(len(selected), 25)
        self.assertFalse(set(selected) & set(range(1, 101)))

    def test_ensure_image_replaces_hash_mismatch(self) -> None:
        payload = b"fixed-image"
        expected = MODULE.hashlib.sha256(payload).hexdigest()
        calls = []

        def downloader(url: str, destination: Path) -> None:
            calls.append(url)
            destination.write_bytes(payload)

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            stale = cache / "000000000001.jpg"
            stale.write_bytes(b"stale")
            path, actual = MODULE.ensure_image(
                profile(1, (0,)), cache, expected_hash=expected, downloader=downloader
            )
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(actual, expected)
        self.assertEqual(len(calls), 1)

    def test_contract_mismatch_is_rejected(self) -> None:
        expected = {"selected_images": [{"image_id": 1}]}
        actual = {"selected_images": [{"image_id": 2}]}
        with self.assertRaisesRegex(RuntimeError, "position 0"):
            MODULE.verify_contract(expected, actual)


if __name__ == "__main__":
    unittest.main()
