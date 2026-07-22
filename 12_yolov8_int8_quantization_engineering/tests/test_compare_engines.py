import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import compare_engines as compare


class ReferenceReuseTests(unittest.TestCase):
    def make_fixture(self, root: Path):
        manifest = root / "manifest.json"
        manifest.write_text('{"fixture": true}\n', encoding="utf-8")
        weights = root / "weights.pt"
        fp32 = root / "fp32.engine"
        fp16 = root / "fp16.engine"
        for path, value in ((weights, b"weights"), (fp32, b"fp32"), (fp16, b"fp16")):
            path.write_bytes(value)
        args = argparse.Namespace(
            reference_report=root / "reference.json",
            manifest=manifest,
            weights=weights,
            fp32_engine=fp32,
            fp16_engine=fp16,
            warmup=3,
        )
        quality_contract = root / "quality_contract.json"
        quality_contract.write_text(json.dumps({
            "schema_version": 1,
            "dataset_manifest_id": "fixture-v1",
            "validation_dataset_id": "fixture-validation-v1",
            "input_shape": [1, 3, 640, 640],
            "evaluation": {
                "confidence_threshold": 0.001,
                "nms_iou_threshold": 0.7,
                "max_detections": 300,
                "metric": "fixture-metric-v1",
            },
            "maximum_drop_from_pytorch": {
                "map50_95": 0.02,
                "map50": 0.02,
                "precision": 0.03,
                "recall": 0.03,
            },
        }), encoding="utf-8")
        experiments = root / "experiments.json"
        experiments.write_text('{"schema_version": 1, "stages": [{"id": "fixture"}]}')
        args.quality_contract = quality_contract
        args.quality_contract_document = compare.load_quality_contract(quality_contract)
        args.experiments = experiments
        compare.configure_quality_contract(args)
        input_shape = (1, 3, 640, 640)
        report = {
            "schema_version": 1,
            "dataset": {
                "dataset_id": "fixture-v1",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            },
            "settings": compare.expected_settings(args, input_shape),
            "quality_contract": compare.contract_identity(
                quality_contract, args.quality_contract_document
            ),
            "experiment": {
                "matrix_sha256": hashlib.sha256(experiments.read_bytes()).hexdigest()
            },
            "regression_thresholds": {
                f"max_{name}_drop": value
                for name, value in compare.regression_thresholds(args).items()
            },
            "artifacts": {
                name: {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for name, path in compare.reference_artifact_paths(args).items()
            },
            "software": compare.current_software(),
            "backends": {
                name: {"passed": True, "metrics": {}}
                for name in ("pytorch", "tensorrt_fp32", "tensorrt_fp16")
            },
        }
        args.reference_report.write_text(json.dumps(report), encoding="utf-8")
        return args, input_shape

    def test_matching_reference_report_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args, input_shape = self.make_fixture(Path(directory))
            report = compare.load_validated_reference_report(
                args, {"dataset_id": "fixture-v1"}, input_shape
            )
            self.assertEqual(report["dataset"]["dataset_id"], "fixture-v1")

    def test_changed_evaluation_setting_rejects_reference_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args, input_shape = self.make_fixture(Path(directory))
            args.quality_contract_document["evaluation"]["confidence_threshold"] = 0.25
            with self.assertRaisesRegex(ValueError, "setting 'confidence' changed"):
                compare.load_validated_reference_report(
                    args, {"dataset_id": "fixture-v1"}, input_shape
                )


if __name__ == "__main__":
    unittest.main()
