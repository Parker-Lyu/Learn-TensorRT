import tempfile
import unittest
from pathlib import Path

from dataset_manifest import DEFAULT_COCO_MANIFEST, REPO_ROOT, build_manifest, load_manifest, resolve_path


class DatasetManifestTests(unittest.TestCase):
    def test_resolve_path_relocates_stale_repository_absolute_path(self) -> None:
        existing = REPO_ROOT / "12_yolov8_int8_quantization_engineering/dataset_manifest.py"
        stale = Path("/stale/container") / existing.relative_to(REPO_ROOT)
        self.assertEqual(resolve_path(Path("unused.json"), str(stale)), existing)

    def test_resolve_path_preserves_unknown_missing_absolute_path(self) -> None:
        missing = Path("/external/dataset/does-not-exist.jpg")
        self.assertEqual(resolve_path(Path("unused.json"), str(missing)), missing)

    def test_default_manifest_is_owned_by_new_lesson(self) -> None:
        expected = Path(__file__).resolve().parents[1] / "data/dataset_manifest.json"
        self.assertEqual(DEFAULT_COCO_MANIFEST, expected)

    def test_overlap_by_content_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration"
            validation = root / "validation"
            labels = root / "labels"
            calibration.mkdir()
            validation.mkdir()
            labels.mkdir()
            (calibration / "a.jpg").write_bytes(b"same-image")
            (validation / "a.jpg").write_bytes(b"same-image")
            (labels / "a.txt").write_text("", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_manifest(
                    calibration, validation, labels, root / "manifest.json", "test-v1"
                )

    def test_hashes_are_checked_when_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration"
            validation = root / "validation"
            labels = root / "labels"
            calibration.mkdir()
            validation.mkdir()
            labels.mkdir()
            (calibration / "cal.jpg").write_bytes(b"calibration")
            (validation / "val.jpg").write_bytes(b"validation")
            (labels / "val.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            document = build_manifest(
                calibration, validation, labels, manifest_path, "test-v1"
            )
            import json
            manifest_path.write_text(json.dumps(document), encoding="utf-8")
            load_manifest(manifest_path)
            (validation / "val.jpg").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_manifest(manifest_path)

    def test_declared_counts_must_match_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "calibration"
            validation = root / "validation"
            labels = root / "labels"
            calibration.mkdir()
            validation.mkdir()
            labels.mkdir()
            (calibration / "cal.jpg").write_bytes(b"calibration")
            (validation / "val.jpg").write_bytes(b"validation")
            (labels / "val.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            document = build_manifest(calibration, validation, labels, manifest_path, "test-v1")
            document["validation_count"] = 2
            import json
            manifest_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation_count"):
                load_manifest(manifest_path)


if __name__ == "__main__":
    unittest.main()
