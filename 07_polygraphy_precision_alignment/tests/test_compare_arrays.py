import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "align_precision.py"
SPEC = importlib.util.spec_from_file_location("align_precision", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ComparisonTests(unittest.TestCase):
    def compare(self, reference, candidate, rtol=0.0, atol=0.0):
        return MODULE.compare_arrays(np.asarray(reference), np.asarray(candidate), rtol, atol)

    def test_integer_subtraction_does_not_wrap(self):
        result = self.compare(np.array([0], dtype=np.uint8), np.array([255], dtype=np.uint8))
        self.assertEqual(result["max_abs_error"], 255.0)
        self.assertFalse(result["allclose"])

    def test_relative_tolerance_uses_reference(self):
        self.assertFalse(self.compare([1.0], [2.0], rtol=0.5)["allclose"])
        self.assertTrue(self.compare([2.0], [1.0], rtol=0.5)["allclose"])

    def test_empty_nonfinite_and_invalid_tolerances(self):
        for values in ([], [np.nan], [np.inf], [-np.inf]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.compare(values, values)
        for tolerance in (-1.0, np.nan, np.inf):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                self.compare([1.0], [1.0], rtol=tolerance)

    def test_rejects_integer_rounding_and_complex_values(self):
        for values in (np.array([2**53 + 1], dtype=np.int64),
                       np.array([2**64 - 1], dtype=np.uint64), [1j]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.compare(values, values)

    def test_float16_subtraction_is_promoted(self):
        result = self.compare(np.array([65504], dtype=np.float16),
                              np.array([-65504], dtype=np.float16))
        self.assertEqual(result["max_abs_error"], 131008.0)

    def test_scalar_and_known_worst_coordinate(self):
        self.assertEqual(self.compare(1.0, 2.0)["max_error_index"], [])
        self.assertEqual(self.compare([[1, 2], [3, 4]], [[1, 2], [3, 9]])["max_error_index"], [1, 1])

    def test_extreme_finite_error_summaries(self):
        for magnitude in (np.nextafter(0.0, 1.0), 1.0e308):
            with self.subTest(magnitude=magnitude):
                result = self.compare(np.zeros(4), np.full(4, magnitude))
                self.assertEqual(result["mean_abs_error"], magnitude)
                self.assertEqual(result["median_abs_error"], magnitude)
                json.dumps(result, allow_nan=False)

    def test_float64_inputs_remain_unchanged(self):
        reference = np.array([1.0, 2.0])
        candidate = np.array([1.5, 2.0])
        reference.setflags(write=False)
        candidate.setflags(write=False)
        self.compare(reference, candidate)
        np.testing.assert_array_equal(reference, [1.0, 2.0])
        np.testing.assert_array_equal(candidate, [1.5, 2.0])

    def test_shape_mismatch_can_be_written_as_a_report(self):
        result = self.compare([1], [[1]])
        self.assertFalse(result["shape_match"])
        report = dict(onnx="model.onnx", trt_mode="engine", input_npy="input.npy",
                      input_name="images", scope="unit-test", scope_note="synthetic",
                      onnxrt_outputs="ort.json", comparison=dict(output_name="output0",
                      metrics=result, likely_cause=MODULE.classify_drift(result, "engine")))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "note.md"
            MODULE.write_precision_note(report, path)
            self.assertIn("Shape match: `False`", path.read_text())
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
