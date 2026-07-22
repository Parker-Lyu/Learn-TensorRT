import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "prepare_coco.py"
SPEC = importlib.util.spec_from_file_location("prepare_coco", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


class CocoPreparationTests(unittest.TestCase):
    def test_category_mapping_compacts_non_contiguous_ids(self) -> None:
        categories = [{"id": value} for value in range(1, 81)]
        categories[-1]["id"] = 90
        mapping = PREPARE.category_mapping(categories)
        self.assertEqual(mapping[1], 0)
        self.assertEqual(mapping[90], 79)

    def test_yolo_line_clips_box_to_image(self) -> None:
        image = {"width": 100, "height": 50}
        annotation = {"bbox": [-10, 10, 120, 50]}
        line = PREPARE.yolo_line(annotation, image, 3)
        self.assertEqual(line, "3 0.50000000 0.60000000 1.00000000 0.80000000")

    def test_write_labels_excludes_crowd_and_writes_empty_files(self) -> None:
        categories = [{"id": value} for value in range(1, 81)]
        document = {
            "images": [
                {"id": index, "file_name": f"{index:012d}.jpg", "width": 100, "height": 100}
                for index in range(1, PREPARE.EXPECTED_VAL_IMAGES + 1)
            ],
            "categories": categories,
            "annotations": [
                {"image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "iscrowd": 0},
                {"image_id": 1, "category_id": 2, "bbox": [1, 1, 5, 5], "iscrowd": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            labels = Path(directory)
            count = PREPARE.write_validation_labels(document, labels)
            self.assertEqual(count, 1)
            self.assertEqual(
                (labels / "000000000001.txt").read_text(encoding="utf-8"),
                "0 0.25000000 0.40000000 0.30000000 0.40000000\n",
            )
            self.assertEqual(
                (labels / "000000000002.txt").read_text(encoding="utf-8"), ""
            )

    def test_committed_manifest_defines_shared_validation_data(self) -> None:
        document = PREPARE.load_canonical_document(PREPARE.CANONICAL_MANIFEST)
        validation = PREPARE.validation_records(document)
        self.assertEqual(document["calibration_count"], 0)
        self.assertEqual(len(validation), 5000)
        self.assertTrue(all("label_sha256" in record for record in validation))

    def test_manifest_destination_rejects_path_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "dataset_manifest.json"
            record = {"image": "../../outside.jpg"}
            with self.assertRaisesRegex(ValueError, "escapes"):
                PREPARE.checked_destination(manifest, record, "validation/images")


if __name__ == "__main__":
    unittest.main()
