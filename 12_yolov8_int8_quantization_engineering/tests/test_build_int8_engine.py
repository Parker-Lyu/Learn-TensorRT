import argparse
import json
import tempfile
import unittest
from collections import Counter
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

    def test_detection_head_profile_requires_expected_execution_structure(self) -> None:
        class Tensor:
            def __init__(self, name) -> None:
                self.name = name

        class Layer:
            def __init__(self, name, layer_type, inputs=None) -> None:
                self.name = name
                self.type = layer_type
                self.inputs = list(inputs or [])
                self.outputs = [Tensor(name + "_output")]

            @property
            def num_inputs(self):
                return len(self.inputs)

            @property
            def num_outputs(self):
                return len(self.outputs)

            def get_input(self, index):
                return self.inputs[index]

            def get_output(self, index):
                return self.outputs[index]

        class Network:
            def __init__(self, layers) -> None:
                self.layers = layers
                self.num_layers = len(layers)
                self.outputs = [layers[-1].outputs[0]]
                self.num_outputs = 1

            def get_layer(self, index):
                return self.layers[index]

            def get_output(self, index):
                return self.outputs[index]

        layers = []
        index = 0
        for layer_type, count in build.DETECTION_HEAD_TOWER_EXPECTED_TYPES.items():
            for _ in range(count):
                branch = "cv2" if index % 2 == 0 else "cv3"
                layers.append(Layer(f"/model.22/{branch}.fixture.{index}", layer_type))
                index += 1
        remaining = Counter(build.DETECTION_HEAD_EXPECTED_TYPES)
        remaining.subtract(build.DETECTION_HEAD_TOWER_EXPECTED_TYPES)
        previous = layers[0].outputs[0]
        for layer_type, count in remaining.items():
            for _ in range(count):
                layer = Layer(f"/model.22/post.fixture.{index}", layer_type, [previous])
                layers.append(layer)
                previous = layer.outputs[0]
                index += 1
        selected = build.precision_profile_layers(Network(layers), "detection_head_fp16")
        self.assertEqual(len(selected), 67)

        layers.pop()
        with self.assertRaisesRegex(ValueError, "unexpected complete-head structure"):
            build.precision_profile_layers(Network(layers), "detection_head_fp16")

    def test_detection_head_inspector_rejects_internal_fp32(self) -> None:
        inspector = json.dumps({
            "Layers": [
                {
                    "Name": "/model.22/Sigmoid",
                    "Inputs": [],
                    "Outputs": [{
                        "Name": "/model.22/Sigmoid_output_0",
                        "Format/Datatype": "Row major linear FP32",
                    }],
                }
            ]
        })
        with self.assertRaisesRegex(ValueError, "FP16 verification failed"):
            build.validate_detection_head_inspector(inspector, {"output0"})

    def test_detection_head_inspector_allows_external_fp32_boundary(self) -> None:
        inspector = json.dumps({
            "Layers": [
                {
                    "Name": "/model.22/Concat_3",
                    "Inputs": [],
                    "Outputs": [
                        {"Name": "output0", "Format/Datatype": "Row major linear FP32"}
                    ],
                }
            ]
        })
        result = build.validate_detection_head_inspector(inspector, {"output0"})
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["external_output_conversions"]), 1)

    def test_removed_small_profiles_are_rejected(self) -> None:
        class EmptyNetwork:
            num_layers = 0

            def get_layer(self, index):
                raise AssertionError(index)

        with self.assertRaisesRegex(ValueError, "unsupported precision profile"):
            build.precision_profile_layers(EmptyNetwork(), "box_outputs_fp16")


if __name__ == "__main__":
    unittest.main()
