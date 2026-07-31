import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


VALIDATE = load("validate_config.py")
GENERATE = load("generate_app_config.py")


class DeepStreamConfigTests(unittest.TestCase):
    def test_inference_config(self):
        result = VALIDATE.validate_infer(ROOT / "config/config_infer_primary_yolov8.txt")
        self.assertEqual(result["batch_size"], 2)

    def test_generator_requires_two_sources(self):
        with tempfile.NamedTemporaryFile() as source:
            with self.assertRaisesRegex(ValueError, "two"):
                GENERATE.render([Path(source.name)])


if __name__ == "__main__":
    unittest.main()
