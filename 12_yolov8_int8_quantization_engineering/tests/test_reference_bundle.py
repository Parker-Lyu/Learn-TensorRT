import json
import tempfile
import unittest
from pathlib import Path

from reference_bundle import assert_compatible, load_bundle, reference_id, write_bundle


def identity() -> dict[str, str]:
    return {
        "weights_sha256": "weights",
        "onnx_sha256": "onnx",
        "validation_manifest_sha256": "manifest",
        "quality_contract_sha256": "contract",
        "preprocessing_id": "letterbox-v1",
        "postprocessing_id": "yolov8-decode-nms-v1",
        "metric_id": "course-coco-like-v1",
        "runtime_id": "trt-8.6.1-cuda-12.2-gpu-test",
        "fp32_engine_sha256": "fp32",
        "fp16_engine_sha256": "fp16",
    }


class ReferenceBundleTests(unittest.TestCase):
    def test_identity_is_order_independent(self) -> None:
        values = identity()
        self.assertEqual(reference_id(values), reference_id(dict(reversed(list(values.items())))))

    def test_round_trip_and_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            report.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            bundle_path = root / "bundle.json"
            write_bundle(bundle_path, identity(), report, {"tensorrt": "8.6.1"})
            bundle = load_bundle(bundle_path)
            assert_compatible(bundle, identity())

    def test_changed_runtime_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            report.write_text("{}", encoding="utf-8")
            bundle_path = root / "bundle.json"
            write_bundle(bundle_path, identity(), report, {})
            changed = identity()
            changed["runtime_id"] = "trt-10.14-cuda-13.0-gpu-test"
            with self.assertRaisesRegex(ValueError, "runtime_id"):
                assert_compatible(load_bundle(bundle_path), changed)


if __name__ == "__main__":
    unittest.main()
