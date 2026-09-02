import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "29_cpp_shared_library_python_binding/build"


class PybindModuleTests(unittest.TestCase):
    def test_extension_is_built_and_importable(self):
        candidates = list(MODULE.glob("trt_inference_py*.so"))
        self.assertTrue(candidates, "build the pybind11 extension first")
        spec = importlib.util.spec_from_file_location("trt_inference_py", candidates[0])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, "TensorRtSession"))


if __name__ == "__main__":
    unittest.main()
