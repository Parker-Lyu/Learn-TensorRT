#!/usr/bin/env python3
"""Focused CPU tests for the ModelOpt PTQ data and metadata helpers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


SCRIPT = Path(__file__).resolve().parent / "modelopt_ptq.py"
SPEC = importlib.util.spec_from_file_location("modelopt_ptq", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PTQ = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PTQ)


class ModelOptPtqTests(unittest.TestCase):
    def test_positive_int_rejects_zero(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            PTQ.positive_int("0")

    def test_calibration_batches_stream_complete_and_partial_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for index, shape in enumerate(((8, 12, 3), (12, 8, 3), (5, 5, 3))):
                path = root / f"{index}.png"
                self.assertTrue(cv2.imwrite(str(path), np.full(shape, index * 20, np.uint8)))
                paths.append(path)

            batches = PTQ.CalibrationBatches(paths, batch_size=2, input_shape=(1, 3, 16, 16))
            materialized = list(batches)
            self.assertEqual(len(batches), 2)
            self.assertEqual(
                [batch.shape for batch in materialized],
                [(2, 3, 16, 16), (1, 3, 16, 16)],
            )
            self.assertTrue(all(batch.dtype == np.float32 for batch in materialized))
            self.assertTrue(all(batch.flags.c_contiguous for batch in materialized))

    def test_calibration_records_select_only_calibration_prefix(self) -> None:
        records = [
            {"split": "calibration", "image": "a", "image_sha256": "a" * 64},
            {"split": "calibration", "image": "b", "image_sha256": "b" * 64},
            {"split": "validation", "image": "c", "image_sha256": "c" * 64},
        ]
        with mock.patch.object(PTQ, "load_manifest", return_value={"records": records}):
            selected = PTQ.calibration_records(Path("manifest.json"), limit=1)
        self.assertEqual([record["image"] for record in selected], ["a"])

    def test_calibration_records_reject_excess_limit(self) -> None:
        records = [{"split": "calibration", "image": "a", "image_sha256": "a" * 64}]
        with mock.patch.object(PTQ, "load_manifest", return_value={"records": records}):
            with self.assertRaisesRegex(ValueError, "manifest contains 1"):
                PTQ.calibration_records(Path("manifest.json"), limit=2)

    def test_write_metadata_marks_smoke_invalid_for_accuracy_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.pt"
            manifest = root / "manifest.json"
            onnx_path = root / "model.onnx"
            metadata_path = root / "model.onnx.json"
            weights.write_bytes(b"weights")
            onnx_path.write_bytes(b"onnx")
            manifest.write_text("{}", encoding="utf-8")
            records = [{"image_sha256": "a" * 64}]
            with mock.patch.object(
                PTQ, "load_manifest", return_value={"dataset_id": "test-calibration-v1"}
            ), mock.patch.object(PTQ, "package_versions", return_value={"test": "1"}):
                PTQ.write_metadata(
                    metadata_path,
                    weights,
                    manifest,
                    records,
                    onnx_path,
                    {"checker_passed": True},
                    1,
                    "smoke",
                )
            document = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(document["candidate_kind"], "smoke")
            self.assertFalse(document["valid_for_accuracy_gate"])
            self.assertEqual(document["preprocess"], PTQ.PREPROCESS_ID)


if __name__ == "__main__":
    unittest.main()
