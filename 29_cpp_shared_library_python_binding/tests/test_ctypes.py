import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "29_cpp_shared_library_python_binding/python"))
from trt_ctypes import TensorRtSession


class BindingTests(unittest.TestCase):
    def test_real_inference_and_error_boundary(self):
        library = ROOT / "29_cpp_shared_library_python_binding/build/libtrt_inference.so"
        engine = ROOT / "17_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine"
        with TensorRtSession(library, engine) as session:
            result = session.infer(np.zeros((1, 3, 640, 640), np.float32))
            self.assertEqual(result.output_elements, 84 * 8400)
            with self.assertRaisesRegex(RuntimeError, "invalid batch"):
                session.infer(np.zeros((5, 3, 640, 640), np.float32))


if __name__ == "__main__": unittest.main()
