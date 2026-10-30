#!/usr/bin/env python3
"""Verify that the TSAN launcher never retries a test failure or a sanitizer report."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest

import run_tsan

LAYOUT = "FATAL: ThreadSanitizer: unexpected memory mapping 0x1234-0x5678\n"
CLANG_LAYOUT = (
    'ThreadSanitizer: CHECK failed: tsan_platform_linux.cpp:282 '
    '"((personality(old_personality | ADDR_NO_RANDOMIZE))) != ((-1))" '
    '(0xffffffffffffffff, 0xffffffffffffffff) (tid=1234)\n'
)


class LauncherTests(unittest.TestCase):
    def fixture(self, cases, expected=None, limit=3):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "child.py"
            script.write_text(
                "import json, os, pathlib, signal, sys\n"
                "root = pathlib.Path(__file__).parent\n"
                "state = root / 'attempts'\n"
                "attempt = int(state.read_text()) if state.exists() else 0\n"
                "cases = json.loads((root / 'cases.json').read_text())\n"
                "state.write_text(str(attempt + 1))\n"
                "case = cases[min(attempt, len(cases)-1)]\n"
                "sys.stdout.write(case.get('stdout', ''))\n"
                "sys.stderr.write(case.get('stderr', ''))\n"
                "if case.get('signal'): os.kill(os.getpid(), case['signal'])\n"
                "sys.exit(case.get('exit', 0))\n"
            )
            (root / "cases.json").write_text(json.dumps(cases))
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = run_tsan.run([sys.executable, str(script)], expected,
                                      max_attempts=limit)
            return result, int((root / "attempts").read_text())

    def test_retries_only_a_recognized_startup_failure(self):
        self.assertEqual(self.fixture([{"exit": 66, "stderr": LAYOUT}, {}]), (0, 2))

    def test_recognizes_clang_personality_failure(self):
        self.assertEqual(self.fixture([
            {"signal": signal.SIGSEGV, "stderr": CLANG_LAYOUT}, {}
        ]), (0, 2))

    def test_does_not_retry_a_data_race(self):
        self.assertEqual(self.fixture([
            {"exit": 66, "stderr": "WARNING: ThreadSanitizer: data race\n"}, {}
        ]), (66, 1))

    def test_does_not_retry_a_lock_report(self):
        self.assertEqual(self.fixture([
            {"exit": 66, "stderr": "WARNING: ThreadSanitizer: lock-order-inversion\n"}, {}
        ]), (66, 1))

    def test_does_not_retry_a_bare_signal(self):
        self.assertEqual(self.fixture([{"signal": signal.SIGTERM}, {}]), (143, 1))

    def test_does_not_retry_a_test_failure_or_mixed_diagnostics(self):
        self.assertEqual(self.fixture([{"exit": 1, "stderr": "assertion failed\n"}, {}]), (1, 1))
        self.assertEqual(self.fixture([
            {"exit": 66, "stderr": LAYOUT + "WARNING: ThreadSanitizer: data race\n"}, {}
        ]), (66, 1))
        self.assertEqual(self.fixture([
            {"exit": 66, "stdout": "main started\n", "stderr": LAYOUT}, {}
        ]), (66, 1))

    def test_exhausted_startup_attempts_fail(self):
        self.assertEqual(self.fixture([{"exit": 66, "stderr": LAYOUT}], limit=2), (1, 2))

    def test_positive_control_requires_the_intended_report(self):
        self.assertEqual(self.fixture([{}], expected="data race"), (1, 1))
        self.assertEqual(self.fixture([
            {"exit": 66, "stderr": "WARNING: ThreadSanitizer: lock-order-inversion\n"}
        ], expected="data race"), (1, 1))
        self.assertEqual(self.fixture([
            {"exit": 66, "stderr": "WARNING: ThreadSanitizer: data race\n"}
        ], expected="data race"), (0, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
