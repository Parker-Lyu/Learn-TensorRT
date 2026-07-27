import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class CaseStudyTests(unittest.TestCase):
    def test_checkpoint_reports_exist(self):
        for name in ("10a_end_to_end_validation.md", "12a_precision_performance.md",
                     "17a_pipeline_performance.md"):
            self.assertTrue((ROOT / "reports" / name).is_file())

    def test_dockerfile_is_multistage(self):
        text = (ROOT / "24_final_portfolio_case_study/Dockerfile").read_text()
        self.assertIn(" AS builder", text)
        self.assertIn("nvcr.io/nvidia/pytorch:25.11-py3", text)
        self.assertIn("nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04", text)
        self.assertIn("COPY --from=builder", text)
        self.assertIn("libnvinfer.so.10", text)
        self.assertIn("assets/img.jpeg", text)
        self.assertNotIn("img2.jpeg", text)

    def test_local_matrix_has_explicit_build_script(self):
        script = ROOT / "24_final_portfolio_case_study/build_local_checks.sh"
        self.assertTrue(script.is_file())
        text = script.read_text()
        self.assertIn("23_cpp_interview_katas", text)
        self.assertIn("CMAKE_BUILD_TYPE=Release", text)

    def test_precision_report_no_longer_uses_smoke_evidence(self):
        text = (ROOT / "reports/12a_precision_performance.md").read_text()
        self.assertIn("5000 fixed", text)
        self.assertNotIn("smoke", text.lower())


if __name__ == "__main__": unittest.main()
