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
        result = VALIDATE.validate_infer(ROOT / "config/config_infer_primary_yolov8_b4.txt")
        self.assertEqual(result["batch_size"], 4)

    def test_generator_requires_two_sources(self):
        with tempfile.NamedTemporaryFile() as source:
            with self.assertRaisesRegex(ValueError, "two"):
                GENERATE.render([Path(source.name)])

    def test_generator_selects_batch_four_inference_config(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = [Path(directory) / f"source-{index}.mp4" for index in range(4)]
            for source in sources:
                source.write_bytes(b"fixture")
            config = GENERATE.render(sources, "../config/config_infer_primary_yolov8_b4.txt")
            self.assertIn("batch-size=4", config)
            self.assertIn("config-file=../config/config_infer_primary_yolov8_b4.txt", config)

    def test_generator_can_enable_osd_and_file_sink(self):
        with tempfile.TemporaryDirectory() as directory:
            sources = [Path(directory) / f"source-{index}.mp4" for index in range(2)]
            for source in sources:
                source.write_bytes(b"fixture")
            output = Path(directory) / "annotated.mp4"
            config = GENERATE.render(sources, render_output=output)
            self.assertIn("enable=1", config)
            self.assertIn("type=3", config)
            self.assertIn(f"output-file={output}", config)


if __name__ == "__main__":
    unittest.main()
