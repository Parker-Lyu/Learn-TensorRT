#!/usr/bin/env python3
"""Focused tests for Engine Inspector role and precision classification."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "inspect_precision.py"
SPEC = importlib.util.spec_from_file_location("inspect_precision", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
INSPECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INSPECT)


class InspectTrt10LayersTests(unittest.TestCase):
    def test_classification_separates_compute_reformat_and_weight_precision(self) -> None:
        data = {
            "Layers": [
                {
                    "Name": "input qdq",
                    "LayerType": "Reformat",
                    "ParameterType": "Reformat",
                    "Origin": "QDQ",
                    "Inputs": [{"Name": "images", "Format/Datatype": "FP32"}],
                    "Outputs": [{"Name": "quantized", "Format/Datatype": "Int8"}],
                },
                {
                    "Name": "conv",
                    "LayerType": "CaskConvolution",
                    "ParameterType": "Convolution",
                    "Weights": {"Type": "Int8"},
                    "Outputs": [{"Name": "conv_out", "Format/Datatype": "FP16"}],
                },
                {
                    "Name": "decode",
                    "LayerType": "PointWiseV2",
                    "ParameterType": "PointWise",
                    "Outputs": [{"Name": "output0", "Format/Datatype": "FP32"}],
                },
            ]
        }
        report = INSPECT.classify(data)
        self.assertEqual(report["total_layers"], 3)
        self.assertEqual(report["reformat_count"], 1)
        self.assertEqual(report["qdq_origin_reformat_count"], 1)
        self.assertEqual(report["fp32_external_boundary_conversion_count"], 1)
        self.assertEqual(report["pure_fp16_compute_count"], 1)
        self.assertEqual(report["pure_fp32_compute_count"], 1)
        self.assertEqual(report["int8_weight_convolutions_by_output_precision"], {"FP16": 1})

    def test_missing_layers_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Layers list"):
            INSPECT.classify({})


if __name__ == "__main__":
    unittest.main()

