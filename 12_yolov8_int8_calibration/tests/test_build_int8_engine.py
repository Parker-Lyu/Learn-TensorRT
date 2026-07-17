import argparse
import json
import tempfile
import unittest
from pathlib import Path

import build_int8_engine as build


class Int8EngineBuildTests(unittest.TestCase):
    def test_cache_identity_separates_calibration_algorithms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            onnx = Path(directory) / "model.onnx"
            onnx.write_bytes(b"fixed-model")
            common = {
                "onnx": onnx,
                "input_shape": (1, 3, 640, 640),
            }
            entropy = argparse.Namespace(**common, calibrator="entropy")
            minmax = argparse.Namespace(**common, calibrator="minmax")
            hashes = ["a" * 64, "b" * 64]
            self.assertNotEqual(build.cache_key(entropy, hashes), build.cache_key(minmax, hashes))

    def test_cache_metadata_records_algorithm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "calibration.cache"
            resources = object.__new__(build.CalibrationResources)
            resources.cache_path = cache
            resources.metadata_path = cache.with_suffix(".cache.json")
            resources.cache_key = "identity"
            resources.algorithm = "minmax"
            resources.write_calibration_cache(b"table")
            metadata = json.loads(resources.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["calibration_algorithm"], "minmax")
            self.assertEqual(metadata["cache_key"], "identity")

    def test_unknown_calibrator_is_rejected_before_allocation(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported calibration algorithm"):
            build.create_calibrator(
                "unknown", [], "images", (1, 3, 640, 640), Path("unused.cache"), "key"
            )

    def test_box_output_profile_constrains_exact_regression_convolutions(self) -> None:
        class Layer:
            def __init__(self, name: str) -> None:
                self.name = name
                self.type = build.trt.LayerType.CONVOLUTION
                self.num_outputs = 1
                self.precision = None
                self.output_types = {}

            def set_output_type(self, index, dtype) -> None:
                self.output_types[index] = dtype

        class Network:
            def __init__(self, layers) -> None:
                self.layers = layers
                self.num_layers = len(layers)

            def get_layer(self, index):
                return self.layers[index]

        class Config:
            def __init__(self) -> None:
                self.flags = []
                self.profiling_verbosity = None

            def set_flag(self, flag) -> None:
                self.flags.append(flag)

        layers = [Layer(name) for name in build.PRECISION_PROFILES["box_outputs_fp16"]]
        config = Config()
        constrained = build.apply_precision_profile(Network(layers), config, "box_outputs_fp16")
        self.assertEqual(tuple(constrained), build.PRECISION_PROFILES["box_outputs_fp16"])
        self.assertEqual(config.flags, [build.trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS])
        for layer in layers:
            self.assertEqual(layer.precision, build.trt.float16)
            self.assertEqual(layer.output_types, {0: build.trt.float16})

    def test_precision_profile_fails_if_an_expected_layer_is_missing(self) -> None:
        class EmptyNetwork:
            num_layers = 0

            def get_layer(self, index):
                raise AssertionError(index)

        class Config:
            def set_flag(self, flag) -> None:
                raise AssertionError(flag)

        with self.assertRaisesRegex(ValueError, "did not match required layer"):
            build.apply_precision_profile(EmptyNetwork(), Config(), "box_outputs_fp16")

    def test_combined_output_profile_has_three_box_and_three_class_layers(self) -> None:
        names = build.PRECISION_PROFILES["box_and_class_outputs_fp16"]
        self.assertEqual(len(names), 6)
        self.assertEqual(sum("/cv2." in name for name in names), 3)
        self.assertEqual(sum("/cv3." in name for name in names), 3)
        self.assertEqual(len(set(names)), 6)

    def test_detection_head_profile_requires_expected_execution_structure(self) -> None:
        class Layer:
            def __init__(self, name, layer_type) -> None:
                self.name = name
                self.type = layer_type

        class Network:
            def __init__(self, layers) -> None:
                self.layers = layers
                self.num_layers = len(layers)

            def get_layer(self, index):
                return self.layers[index]

        layers = []
        index = 0
        for layer_type, count in build.DETECTION_HEAD_EXPECTED_TYPES.items():
            for _ in range(count):
                branch = "cv2" if index % 2 == 0 else "cv3"
                layers.append(Layer(f"/model.22/{branch}.fixture.{index}", layer_type))
                index += 1
        selected = build.precision_profile_layers(Network(layers), "detection_head_fp16")
        self.assertEqual(len(selected), 42)

        layers.pop()
        with self.assertRaisesRegex(ValueError, "unexpected layer structure"):
            build.precision_profile_layers(Network(layers), "detection_head_fp16")


if __name__ == "__main__":
    unittest.main()
