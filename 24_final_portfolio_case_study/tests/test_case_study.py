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
        self.assertIn("FROM nvidia/cuda", text)
        self.assertIn("COPY --from=builder", text)


if __name__ == "__main__": unittest.main()
