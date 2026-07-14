import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("diagnose", ROOT / "diagnose_model.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class DiagnosisTests(unittest.TestCase):
    def test_detects_custom_domain_after_model_is_created(self):
        model = ROOT / "outputs/unsupported_swish.onnx"
        if not model.exists():
            self.skipTest("generate the demo model first")
        result = MODULE.diagnose(model)
        self.assertEqual(result["custom_domain_nodes"][0]["op_type"], "AcmeSwish")


if __name__ == "__main__":
    unittest.main()
