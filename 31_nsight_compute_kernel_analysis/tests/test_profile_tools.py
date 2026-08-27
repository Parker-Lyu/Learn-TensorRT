import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


LESSON = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, LESSON / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROFILE = load_module("lesson31_profile", "profile_kernels.py")
REPORT = load_module("lesson31_report", "generate_report.py")


class CommandTests(unittest.TestCase):
    def test_ncu_command_separates_replay_from_benchmark(self):
        command = PROFILE.ncu_command(
            "ncu", Path("bench"), Path("capture"), Path("target.json"),
            "unfused", warmup=3,
        )
        self.assertIn("--replay-mode", command)
        self.assertEqual(command[command.index("--launch-skip") + 1], "6")
        self.assertEqual(command[command.index("--launch-count") + 1], "2")
        self.assertIn("WarpStateStats", command)

    def test_nsys_command_has_cuda_and_nvtx_trace(self):
        command = PROFILE.nsys_command(
            "nsys", Path("lesson20"), Path("capture"), iterations=3
        )
        self.assertIn("--trace=cuda,nvtx", command)
        self.assertIn("lesson20", command)
        self.assertIn("--iterations", command)


class MetricTests(unittest.TestCase):
    def test_parses_and_filters_ncu_raw_csv(self):
        text = (
            '==PROF== Connected\n'
            '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
            '"0","kernel","launch__registers_per_thread","register/thread","20"\n'
            '"0","kernel","unrelated","byte","4"\n'
        )
        rows = PROFILE.parse_ncu_csv(text)
        summary = PROFILE.summarize_ncu_rows(rows)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["value"], "20")

    def test_rejects_missing_metric_table(self):
        with self.assertRaises(ValueError):
            PROFILE.parse_ncu_csv("not csv")

    def test_parses_current_wide_ncu_csv(self):
        text = (
            '"ID","Kernel Name","launch__registers_per_thread","gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"\n'
            '"","","register/thread","%"\n'
            '"0","kernel","24","55.5"\n'
        )
        rows = PROFILE.parse_ncu_csv(text)
        summary = PROFILE.summarize_ncu_rows(rows)
        self.assertEqual([item["value"] for item in summary], ["24", "55.5"])


class ReportTests(unittest.TestCase):
    @staticmethod
    def results(**p50):
        return [
            {"variant": name, "timing_ms": {"p50": value}}
            for name, value in p50.items()
        ]

    def test_fusion_decision_accepts_measured_kernel_improvement(self):
        results = self.results(
            unfused=2.0,
            baseline_16x16=1.0,
            block_32x8=1.1,
            linear=1.2,
            vectorized=1.3,
        )
        decision, reduction, speedup = REPORT.fusion_decision(results)
        self.assertAlmostEqual(reduction, 50.0)
        self.assertAlmostEqual(speedup, 2.0)
        self.assertIn("accepted", decision)

    def test_alternative_decision_does_not_claim_deployment_win(self):
        results = [
            {"variant": "unfused", "timing_ms": {"p50": 2.0}},
            {"variant": "baseline_16x16", "timing_ms": {"p50": 1.0}},
            {"variant": "block_32x8", "timing_ms": {"p50": 1.1}},
            {"variant": "linear", "timing_ms": {"p50": 0.8}},
            {"variant": "vectorized", "timing_ms": {"p50": 1.2}},
        ]
        decision, improvement = REPORT.alternative_decision(results)
        self.assertAlmostEqual(improvement, 20.0)
        self.assertIn("kernel candidate", decision)

    def test_decisions_require_complete_controlled_variant_set(self):
        results = self.results(unfused=2.0, baseline_16x16=1.0)
        with self.assertRaisesRegex(ValueError, "missing variants"):
            REPORT.fusion_decision(results)

    def test_candidate_table_classifies_each_follow_up(self):
        results = self.results(
            unfused=2.0,
            baseline_16x16=1.0,
            block_32x8=0.9,
            linear=1.0,
            vectorized=1.2,
        )
        table = REPORT.candidate_table(results)
        self.assertIn("block_32x8 | 0.900000 | 10.00% | candidate", table)
        self.assertIn("linear | 1.000000 | 0.00% | inconclusive", table)
        self.assertIn("vectorized | 1.200000 | -20.00% | rejected", table)

    def test_benchmark_validation_rejects_numerical_failure(self):
        data = {
            "schema_version": 1,
            "results": [{
                "variant": "baseline_16x16",
                "maximum_absolute_error": 0.1,
                "timing_ms": {"mean": 1.0, "p50": 1.0, "p90": 1.0},
            }],
        }
        with self.assertRaisesRegex(ValueError, "correctness"):
            REPORT.validate_benchmark(data)

    def test_pipeline_section_uses_lesson21_schema(self):
        metrics = {
            "fps": 42.0,
            "p50_ms": 3.0,
            "p90_ms": 4.0,
            "p99_ms": 5.0,
            "preprocess_ms": 6.0,
            "inference_ms": 7.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.json"
            path.write_text(json.dumps(metrics), encoding="utf-8")
            text = REPORT.pipeline_section({"pipeline": {"available": True, "artifact": str(path)}})
        self.assertIn("42.000", text)
        self.assertIn("P50/P90/P99", text)
        self.assertIn("context only", text)


if __name__ == "__main__":
    unittest.main()
