import json
import tempfile
import unittest
from pathlib import Path

from experiment_contract import validate_engine_for_experiment
from quality_contract import evaluation_settings, load_quality_contract, regression_thresholds


class ContractTests(unittest.TestCase):
    def test_quality_contract_is_the_executable_settings_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality.json"
            path.write_text(json.dumps({
                "schema_version": 2,
                "dataset_manifest_id": "dataset-v1",
                "validation_dataset_id": "validation-v1",
                "input_shape": [1, 3, 320, 320],
                "evaluation": {
                    "confidence_threshold": 0.02,
                    "nms_iou_threshold": 0.6,
                    "max_detections": 100,
                    "metric": "metric-v1",
                },
                "baseline_gate": {"reference": "pytorch_fp32", "maximum_drop": {
                    "map50_95": 0.01, "map50": 0.02, "precision": 0.03, "recall": 0.04}},
                "int8_gate": {"references": ["pytorch_fp32", "tensorrt_fp16"], "maximum_drop": {
                    "map50_95": 0.01, "map50": 0.02, "precision": 0.03, "recall": 0.04}},
            }), encoding="utf-8")
            contract = load_quality_contract(path)
            self.assertEqual(evaluation_settings(contract)["input_shape"], [1, 3, 320, 320])
            self.assertEqual(regression_thresholds(contract)["recall"], 0.04)

    def test_experiment_validation_rejects_swapped_engine_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "candidate.engine"
            engine.write_bytes(b"engine")
            metadata = root / "candidate.engine.json"
            import hashlib
            metadata.write_text(json.dumps({
                "schema_version": 1,
                "tensorrt_version": "10.14.1.48",
                "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
                "calibration_algorithm": "unexpected",
            }), encoding="utf-8")
            experiments = root / "experiments.json"
            experiments.write_text(json.dumps({
                "schema_version": 1,
                "stages": [{
                    "id": "minmax",
                    "runtime": "tensorrt_10_14",
                    "engine_metadata": {"calibration_algorithm": "minmax"},
                }],
            }), encoding="utf-8")
            environments = root / "environments.json"
            environments.write_text(json.dumps({
                "environments": {"tensorrt_10_14": {"tensorrt": "10.14.1.48"}}
            }), encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                validate_engine_for_experiment(
                    "minmax", experiments, environments, engine, metadata, manifest, "10.14.1.48"
                )


if __name__ == "__main__":
    unittest.main()
