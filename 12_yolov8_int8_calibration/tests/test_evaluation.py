import tempfile
import unittest
from pathlib import Path

from evaluation import allocate_prediction_buffer, detection_metrics, load_yolo_labels


class EvaluationTests(unittest.TestCase):
    def test_perfect_detection_scores_one(self) -> None:
        truth = {"image": [{"class_id": 0, "box_xyxy": [10.0, 10.0, 30.0, 30.0]}]}
        predictions = {
            "image": [{"class_id": 0, "confidence": 0.9, "box_xyxy": [10.0, 10.0, 30.0, 30.0]}]
        }
        metrics = detection_metrics(predictions, truth)
        for value in metrics.values():
            self.assertAlmostEqual(value, 1.0)

    def test_invalid_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            path.write_text("0 1.2 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_yolo_labels(path, 100, 100)

    def test_zero_size_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "label.txt"
            path.write_text("0 0.5 0.5 0.0 0.2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_yolo_labels(path, 100, 100)

    def test_prediction_for_absent_class_counts_as_false_positive(self) -> None:
        truth = {"image": [{"class_id": 0, "box_xyxy": [10.0, 10.0, 30.0, 30.0]}]}
        predictions = {"image": [
            {"class_id": 0, "confidence": 0.9, "box_xyxy": [10.0, 10.0, 30.0, 30.0]},
            {"class_id": 1, "confidence": 0.8, "box_xyxy": [40.0, 40.0, 50.0, 50.0]},
        ]}
        metrics = detection_metrics(predictions, truth)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 1.0)

    def test_prediction_buffer_has_bounded_compact_capacity(self) -> None:
        buffer = allocate_prediction_buffer(5000, 300)
        self.assertEqual(len(buffer), 1_500_000)
        self.assertLessEqual(buffer.dtype.itemsize, 32)


if __name__ == "__main__":
    unittest.main()
