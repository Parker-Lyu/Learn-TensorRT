import importlib.util
import sys
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
    def test_ncu_filters_owned_kernels_and_skips_warmup(self):
        command = PROFILE.ncu_command(
            "ncu", Path("bench"), Path("capture"), Path("target.json"),
            "baseline", warmup=3,
        )
        self.assertEqual(command[command.index("--launch-skip") + 1], "6")
        self.assertEqual(command[command.index("--launch-count") + 1], "2")
        self.assertEqual(command[command.index("--kernel-name") + 1],
                         "regex:layer_norm_.*_kernel")
        self.assertEqual(command[command.index("--scope") + 1], "layernorm")
        self.assertIn("WarpStateStats", command)

    def test_counter_permission_error_is_classified(self):
        result = PROFILE.CommandResult([], 1, 0.1, "ERR_NVGPUCTRPERM", "")
        self.assertTrue(PROFILE.is_counter_permission_error(result))
        other = PROFILE.CommandResult([], 1, 0.1, "other failure", "")
        self.assertFalse(PROFILE.is_counter_permission_error(other))

    def test_nsys_profiles_the_complete_mlp_with_nvtx(self):
        command = PROFILE.nsys_command(
            "nsys", Path("mlp"), Path("capture"), Path("target.json"), 2, 4
        )
        self.assertIn("--trace=cuda,nvtx,cublas", command)
        self.assertIn("mlp", command)
        self.assertEqual(command[command.index("--iterations") + 1], "4")
        self.assertEqual(command[command.index("--scope") + 1], "network")


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
            '"ID","Kernel Name","launch__registers_per_thread",'
            '"gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"\n'
            '"","","register/thread","%"\n'
            '"0","kernel","24","55.5"\n'
        )
        rows = PROFILE.parse_ncu_csv(text)
        summary = PROFILE.summarize_ncu_rows(rows)
        self.assertEqual([item["value"] for item in summary], ["24", "55.5"])


class ReportTests(unittest.TestCase):
    @staticmethod
    def result(variant: str, layernorm: float, network: float, error: float = 0.0):
        timing = lambda value: {"mean": value, "p50": value, "p90": value}
        return {
            "variant": variant,
            "layernorm_timing_ms": timing(layernorm),
            "network_timing_ms": timing(network),
            "maximum_absolute_error": error,
        }

    def test_accepts_only_when_operator_and_network_improve(self):
        decision = REPORT.optimization_decision([
            self.result("baseline", 2.0, 8.0),
            self.result("fused", 1.0, 7.0),
        ])
        self.assertAlmostEqual(decision["operator_reduction"], 50.0)
        self.assertAlmostEqual(decision["network_reduction"], 12.5)
        self.assertIn("accepted", decision["conclusion"])

    def test_rejects_kernel_only_win_for_workload(self):
        decision = REPORT.optimization_decision([
            self.result("baseline", 2.0, 8.0),
            self.result("fused", 1.0, 8.5),
        ])
        self.assertIn("kernel optimization succeeds", decision["conclusion"])
        self.assertIn("rejected for this workload", decision["conclusion"])

    def test_validation_requires_both_variants(self):
        data = {"schema_version": 2, "results": [self.result("baseline", 1.0, 2.0)]}
        with self.assertRaisesRegex(ValueError, "missing variants"):
            REPORT.validate_benchmark(data)

    def test_validation_rejects_numerical_failure(self):
        data = {"schema_version": 2, "results": [
            self.result("baseline", 1.0, 2.0),
            self.result("fused", 0.8, 1.8, error=0.1),
        ]}
        with self.assertRaisesRegex(ValueError, "correctness"):
            REPORT.validate_benchmark(data)


if __name__ == "__main__":
    unittest.main()
