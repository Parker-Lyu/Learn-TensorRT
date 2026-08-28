import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "32_final_portfolio_case_study/generate_case_study.py"
SPEC = importlib.util.spec_from_file_location("generate_case_study", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class CaseStudyTests(unittest.TestCase):
    def test_schema_v3_pipeline_tables_are_parsed_by_heading_and_column(self):
        report = """## Real Integrated Load Matrix

| Batch | Completed | FPS | P50 ms | P90 ms | P99 ms | Queue peak |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 43.196 | 44.779 | 44.779 | 44.779 | 2 |

## Overload and Freshness Policies

| Policy | Captured | Completed | Evicted | Aborted | Queue peak | FPS | P99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| block | 200 | 200 | 0 | 0 | 4 | 367.457 | 63.499 |
"""
        load = MODULE.markdown_table(report, "Real Integrated Load Matrix")
        policies = MODULE.markdown_table(report, "Overload and Freshness Policies")

        self.assertEqual("43.196", load[0]["FPS"])
        self.assertEqual("200", policies[0]["Completed"])

    def test_checkpoint_reports_are_generated_inputs_not_tracked_fixtures(self):
        ignore_lines = (ROOT / ".gitignore").read_text().splitlines()
        self.assertIn("reports/", ignore_lines)
        source = MODULE_PATH.read_text()
        for name in ("12_end_to_end_validation.md", "15_precision_performance.md",
                     "22_pipeline_performance.md"):
            self.assertIn(f'ROOT / "reports/{name}"', source)

    def test_dockerfile_is_multistage(self):
        text = (ROOT / "32_final_portfolio_case_study/Dockerfile").read_text()
        self.assertIn(" AS builder", text)
        self.assertIn("nvcr.io/nvidia/pytorch:25.11-py3", text)
        self.assertIn("nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04", text)
        self.assertIn("COPY --from=builder", text)
        self.assertIn("libnvinfer.so.10", text)
        self.assertIn("assets/img.jpeg", text)
        self.assertIn("yolov8n_static_fp16_strong.engine", text)
        dockerignore = (ROOT / ".dockerignore").read_text()
        self.assertIn("!32_final_portfolio_case_study/outputs/yolov8n_static_fp16_strong.engine", dockerignore)

    def test_local_matrix_has_explicit_build_script(self):
        script = ROOT / "32_final_portfolio_case_study/build_local_checks.sh"
        self.assertTrue(script.is_file())
        text = script.read_text()
        self.assertIn("31_nsight_compute_kernel_analysis", text)
        self.assertIn("CMAKE_BUILD_TYPE=Release", text)

    def test_case_study_uses_quality_and_performance_for_precision_choice(self):
        decision, next_step = MODULE.choose_precision(
            {"FP32": "PASS", "FP16": "PASS", "INT8": "PASS"},
            fp16_throughput=600.0,
            int8_throughput=500.0,
        )
        self.assertIn("FP16 is the current deployment choice", decision)
        self.assertIn("INT8 passes the declared quality gate", decision)
        self.assertIn("performance regression", next_step)


if __name__ == "__main__": unittest.main()
