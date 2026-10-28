"""CPU tests for conversion gates, cache invalidation, and benchmark evidence."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

LESSON = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, LESSON / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BUILD = load_module("lesson06_build", "build_and_benchmark.py")
SUMMARY = load_module("lesson06_summary", "summarize_results.py")
PREPARE = load_module("lesson06_prepare", "prepare_fp16_onnx.py")

# Synthetic log values deliberately distinguish total latency, copies, and GPU compute.
LOG = "\n".join([
    "[I] Throughput: 2.5e+2 qps",
    "[I] [TRT] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +19, now: CPU 0, GPU 37 (MiB)",
    *[f"[I] {label}: min = {value} ms, max = {value + 1} ms, mean = {value + .5} ms, median = {value + .4} ms, "
      f"percentile(50%) = {value + .4} ms, percentile(90%) = {value + .7} ms, percentile(95%) = {value + .8} ms, percentile(99%) = {value + .9} ms"
      for label, value in (("H2D Latency", 1), ("D2H Latency", 2), ("Latency", 10),
                           ("GPU Compute Time", 5), ("Enqueue Time", 3))],
    "&&&& PASSED TensorRT.trtexec",
])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        printer = patch.object(BUILD, "print", create=True)
        printer.start()
        self.addCleanup(printer.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.onnx"
        self.converted = self.root / "converted.onnx"
        self.source.write_bytes(b"source graph")
        self.converted.write_bytes(b"converted graph")
        self.args = BUILD.parse_args([
            "--onnx", str(self.source), "--dynamic-onnx", str(self.source),
            "--static-autocast-onnx", str(self.converted),
            "--dynamic-autocast-onnx", str(self.converted),
            "--output-dir", str(self.root / "engines"), "--builds", "static_fp16_strong",
        ])
        self.write_gate()
        self.environment = {"tensorrt": "test-version", "gpu_and_driver": "test-gpu"}

    def write_gate(self, **changes):
        report = {"comparison": {"passed": True}, "source_sha256": BUILD.sha256(self.source),
                  "converted_sha256": BUILD.sha256(self.converted)}
        report.update(changes)
        for kind in ("static", "dynamic"):
            (self.root / f"{kind}_fp16_onnx_validation.json").write_text(json.dumps(report))

    def build(self):
        return BUILD.planned(self.args)[0]

    def test_report_is_resolved_next_to_graph_even_with_separate_engine_directory(self):
        self.assertEqual(self.build().validation_report.parent, self.converted.parent)

    def test_missing_failed_and_malformed_reports_block_planning(self):
        path = self.root / "static_fp16_onnx_validation.json"
        for text in (None, '{"comparison":{"passed":false}}', '{"comparison":{"passed":"true"}}', "invalid", "null", "[]"):
            with self.subTest(text=text):
                if text is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_text(text)
                with self.assertRaises((FileNotFoundError, ValueError)):
                    self.build()

    def test_source_and_converted_hashes_are_both_enforced(self):
        for path in (self.source, self.converted):
            original = path.read_bytes()
            path.write_bytes(b"changed graph")
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "hash differs"):
                self.build()
            path.write_bytes(original)

    def test_only_selected_models_are_required_and_conflicts_are_rejected(self):
        self.args.builds = ["static_fp32"]
        self.converted.unlink()
        self.assertEqual(self.build().name, "static_fp32")
        self.args.builds = ["dynamic_fp16_legacy"]
        self.args.skip_dynamic = True
        with self.assertRaisesRegex(ValueError, "disabled"):
            self.build()

    def test_precision_and_profile_commands(self):
        self.args.builds = None
        builds = {item.name: item for item in BUILD.planned(self.args)}
        self.assertIn("--noTF32", BUILD.command(builds["static_fp32"], self.args))
        self.assertIn("--fp16", BUILD.command(builds["static_fp16_legacy"], self.args))
        strong = BUILD.command(builds["dynamic_fp16_strong"], self.args)
        self.assertIn("--stronglyTyped", strong)
        self.assertNotIn("--fp16", strong)
        self.assertIn("--minShapes=images:1x3x320x320", strong)
        loaded = BUILD.command(builds["dynamic_fp16_strong"], self.args, reuse_engine=True)
        self.assertIn("--shapes=images:1x3x640x640", loaded)
        self.assertTrue(any(item.startswith("--loadEngine=") for item in loaded))
        self.assertFalse(any(item.startswith("--onnx=") or item.startswith("--minShapes=") for item in loaded))

    def test_explicit_timing_cache(self):
        self.args.timing_cache_file = self.root / "existing.cache"
        self.assertIn(f"--timingCacheFile={self.args.timing_cache_file}", BUILD.command(self.build(), self.args))

    def fake_trtexec(self, build, write_artifacts=True):
        def invoke(cmd, stdout, **kwargs):
            stdout.write(LOG)
            if write_artifacts:
                build.engine_path.write_bytes(b"engine")
                for path in (build.times_path, build.layer_info_path, build.profile_path):
                    path.write_text('{"Layers":[{}]}' if path == build.layer_info_path else "[{}]")
            return SimpleNamespace(returncode=0)
        return invoke

    def test_valid_receipt_reuses_engine_but_runs_fresh_measurement(self):
        build = self.build()
        with patch.object(BUILD.subprocess, "run", side_effect=self.fake_trtexec(build)) as runner:
            self.assertFalse(BUILD.run(build, self.args, self.environment)["reused_engine"])
            self.assertTrue(BUILD.run(build, self.args, self.environment)["reused_engine"])
            self.assertEqual(runner.call_count, 2)
            self.assertIn(f"--loadEngine={build.engine_path}", runner.call_args.args[0])
        self.args.rebuild = True
        with patch.object(BUILD.subprocess, "run", side_effect=self.fake_trtexec(build)):
            self.assertFalse(BUILD.run(build, self.args, self.environment)["reused_engine"])

    def test_model_environment_settings_and_engine_corruption_invalidate_receipt(self):
        build = self.build()
        with patch.object(BUILD.subprocess, "run", side_effect=self.fake_trtexec(build)):
            BUILD.run(build, self.args, self.environment)
        signature = BUILD.engine_signature(build, self.args, self.environment)
        self.assertTrue(BUILD.reusable_engine(build, signature))
        for changed in ({**signature, "onnx_sha256": "other"},
                        {**signature, "environment": {"gpu_and_driver": "other"}},
                        {**signature, "workspace_mib": 1}, {**signature, "max_shapes": "other"}):
            self.assertFalse(BUILD.reusable_engine(build, changed))
        build.engine_path.write_bytes(b"corrupted")
        self.assertFalse(BUILD.reusable_engine(build, signature))

    def test_malformed_receipts_require_rebuild(self):
        build = self.build()
        build.engine_path.parent.mkdir()
        build.engine_path.write_bytes(b"engine")
        signature = BUILD.engine_signature(build, self.args, self.environment)
        for content in ("null", "[]", "invalid", "{}"):
            build.engine_path.with_suffix(".build.json").write_text(content)
            with self.subTest(content=content):
                self.assertFalse(BUILD.reusable_engine(build, signature))

    def test_failed_conversion_blocks_cached_engine_too(self):
        build = self.build()
        self.write_gate(comparison={"passed": False})
        with patch.object(BUILD.subprocess, "run") as runner:
            with self.assertRaises(ValueError):
                BUILD.run(build, self.args, self.environment)
            runner.assert_not_called()

    def test_success_exit_without_new_artifacts_cannot_inherit_stale_exports(self):
        build = self.build()
        build.engine_path.parent.mkdir()
        for path in (build.engine_path, build.times_path, build.layer_info_path, build.profile_path):
            path.write_text("[{}]")
        with patch.object(BUILD.subprocess, "run", side_effect=self.fake_trtexec(build, False)):
            with self.assertRaisesRegex(RuntimeError, "Missing or empty"):
                BUILD.run(build, self.args, self.environment)
        self.assertFalse(build.engine_path.with_suffix(".build.json").exists())


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.log = self.root / "trtexec.log"
        self.log.write_text(LOG)

    def test_percentiles_are_scoped_to_their_metric(self):
        metrics = SUMMARY.extract_metrics(self.log)
        self.assertEqual(metrics["latency_ms"]["percentiles"]["99"], 10.9)
        self.assertEqual(metrics["gpu_compute_ms"]["percentiles"]["99"], 5.9)
        self.assertEqual(metrics["h2d_ms"]["mean"], 1.5)
        self.assertEqual(metrics["context_gpu_mib"], 37)
        self.assertEqual(metrics["throughput_qps"], 250)

    def record(self):
        record = {"name": "dynamic_fp16_strong", "precision": "fp16", "log": str(self.log)}
        for key in ("engine", "times", "layers", "profile"):
            path = self.root / key
            path.write_text('{"Layers":[{}]}' if key == "layers" else "[{}]")
            record[key] = str(path)
        return record

    def test_missing_empty_corrupt_and_failed_evidence_are_rejected(self):
        record = self.record()
        SUMMARY.verified_metrics(record)
        for content in (None, "", "invalid json", "[]", "null"):
            path = Path(record["times"])
            if content is None:
                path.unlink()
            else:
                path.write_text(content)
            with self.subTest(content=content), self.assertRaises(ValueError):
                SUMMARY.verified_metrics(record)
        path.write_text("[{}]")
        self.log.write_text(LOG.replace("&&&& PASSED", "&&&& FAILED"))
        with self.assertRaises(ValueError):
            SUMMARY.verified_metrics(record)
        self.log.write_text(LOG.replace("percentile(99%)", "unparsed_percentile"))
        with self.assertRaisesRegex(ValueError, "missing statistics"):
            SUMMARY.verified_metrics(record)

    def test_report_contains_profiles_memory_and_separate_tail_metrics(self):
        record = self.record()
        record.update(typing_mode="strongly_typed", shapes="images:1x3x640x640",
                      min_shapes="images:1x3x320x320", opt_shapes="images:1x3x640x640",
                      max_shapes="images:4x3x640x640", command=["trtexec", "--loadEngine=model.engine"])
        report = SUMMARY.build_report(dict(runtime_environment={"tensorrt": "test"},
            workspace_mib=2048, warmup_ms=500, duration_sec=5, avg_runs=10, builds=[record]))
        for expected in ("images:4x3x640x640", "TRT context GPU MiB", "P99 ms", "10.900", "5.900", "--loadEngine=model.engine"):
            self.assertIn(expected, report)

    def test_custom_output_directory_does_not_force_default_report_path(self):
        self.assertIsNone(SUMMARY.parse_args(["--output-dir", str(self.root)]).report)


class ConversionTests(unittest.TestCase):
    def test_invalid_tolerance_empty_shape_and_nonfinite_outputs(self):
        reference = np.zeros((1, 84, 2), dtype=np.float32)
        for tolerance in (-1, np.nan, np.inf):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                PREPARE.compare_outputs([reference], [reference], tolerance, 0)
        for value in (np.empty((0,)), np.array([np.nan]), np.array([np.inf])):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PREPARE.compare_outputs([value], [value], 0, 0)
        with self.assertRaises(ValueError):
            PREPARE.compare_outputs([], [], 0, 0)
        with self.assertRaises(ValueError):
            PREPARE.compare_outputs([reference], [reference[:, :, :1]], 0, 0)

    def test_coordinates_and_scores_have_distinct_tolerances(self):
        reference = np.ones((1, 84, 2), dtype=np.float32)
        reference[:, :4, :] = 100
        candidate = reference.copy()
        candidate[:, :4, :] += 1
        result = PREPARE.compare_yolov8_raw_outputs([reference], [candidate], .05, .05, .01, .02)
        self.assertTrue(result["passed"])
        candidate[:, 4:, :] += .1
        self.assertFalse(PREPARE.compare_yolov8_raw_outputs([reference], [candidate], .05, .05, .01, .02)["passed"])


if __name__ == "__main__":
    unittest.main()
