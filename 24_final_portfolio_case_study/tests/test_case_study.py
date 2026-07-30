import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "24_final_portfolio_case_study/generate_case_study.py"
SPEC = importlib.util.spec_from_file_location("generate_case_study", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


class CaseStudyTests(unittest.TestCase):
    def test_checkpoint_reports_are_generated_inputs_not_tracked_fixtures(self):
        ignore_lines = (ROOT / ".gitignore").read_text().splitlines()
        self.assertIn("reports/", ignore_lines)
        source = MODULE_PATH.read_text()
        for name in ("10a_end_to_end_validation.md", "12a_precision_performance.md",
                     "17a_pipeline_performance.md"):
            self.assertIn(f'ROOT / "reports/{name}"', source)

    def test_dockerfile_is_multistage(self):
        text = (ROOT / "24_final_portfolio_case_study/Dockerfile").read_text()
        self.assertIn(" AS builder", text)
        self.assertIn("nvcr.io/nvidia/pytorch:25.11-py3", text)
        self.assertIn("nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04", text)
        self.assertIn("COPY --from=builder", text)
        self.assertIn("libnvinfer.so.10", text)
        self.assertIn("assets/img.jpeg", text)
        self.assertIn("yolov8n_static_autocast_fp16.engine", text)

    def test_local_matrix_has_explicit_build_script(self):
        script = ROOT / "24_final_portfolio_case_study/build_local_checks.sh"
        self.assertTrue(script.is_file())
        text = script.read_text()
        self.assertIn("23_cpp_interview_katas", text)
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
