#!/usr/bin/env python3
"""CPU tests for Step 06 unlabeled output validation."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPT = Path(__file__).resolve().parent / "validate_outputs.py"
with mock.patch.dict("sys.modules", {"compare_engines": mock.MagicMock()}):
    SPEC = importlib.util.spec_from_file_location("validate_outputs", SCRIPT)
    assert SPEC is not None and SPEC.loader is not None
    VALIDATE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(VALIDATE)


class ValidateTrt10OutputsTests(unittest.TestCase):
    def test_valid_output_and_drift_aggregation(self) -> None:
        output = np.zeros((1, 84, 8400), dtype=np.float32)
        output[:, 4, 0] = 0.5
        summary = VALIDATE.validate_output(output)
        self.assertEqual(summary["shape"], [1, 84, 8400])
        aggregate = VALIDATE.aggregate_drifts(
            [
                {"max_abs": 1.0, "mean_abs": 0.1, "p99_abs": 0.2},
                {"max_abs": 2.0, "mean_abs": 0.05, "p99_abs": 0.3},
            ]
        )
        self.assertEqual(aggregate, {"max_abs": 2.0, "mean_abs": 0.1, "p99_abs": 0.3})

    def test_non_finite_and_collapsed_scores_are_rejected(self) -> None:
        output = np.zeros((1, 84, 8400), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "collapsed"):
            VALIDATE.validate_output(output)
        output[:, 4, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            VALIDATE.validate_output(output)


if __name__ == "__main__":
    unittest.main()

