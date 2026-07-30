import argparse
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS_DIR))
import verify_preprocessing_parity as parity  # noqa: E402


class PreprocessingParityTests(unittest.TestCase):
    def test_synthetic_aspect_ratios_are_byte_identical(self) -> None:
        # Odd dimensions and extreme aspect ratios exercise resize rounding and asymmetric padding.
        sizes = [(1, 1), (7, 13), (13, 7), (31, 1920), (1080, 33)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (height, width) in enumerate(sizes):
                values = np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3)
                image = ((values * 37 + index * 19) % 256).astype(np.uint8)
                path = root / f"case_{index}_{width}x{height}.png"
                self.assertTrue(cv2.imwrite(str(path), image))
                with self.subTest(width=width, height=height):
                    result = parity.compare_image(path, (1, 3, 640, 640))
                    self.assertTrue(result["byte_identical"])
                    self.assertEqual(result["max_abs_difference"], 0.0)
                    self.assertEqual(result["differing_values"], 0)

    def test_shape_parser_rejects_non_single_rgb_input(self) -> None:
        for value in ("2x3x640x640", "1x1x640x640", "1x3x0x640", "bad"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parity.parse_shape(value)


if __name__ == "__main__":
    unittest.main()
