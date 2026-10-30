#!/usr/bin/env python3
"""Require the intended injected failure and reconciled terminal metrics, not just a nonzero exit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    executable, engine, image, variable, value, diagnostic = sys.argv[1:]
    with tempfile.TemporaryDirectory(prefix="lesson21-failure-") as directory:
        environment = os.environ.copy()
        environment[variable] = value
        result = subprocess.run(
            [executable, engine, image, "32", "4", "2", directory],
            env=environment, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 1, result.stdout + result.stderr
        assert diagnostic in result.stderr, result.stderr
        metrics = json.loads((Path(directory) / "metrics.json").read_text())
        assert metrics["normal_drain"] is False
        assert metrics["captured"] == metrics["admitted"] + metrics["rejected_on_admission"]
        assert metrics["admitted"] == (
            metrics["evicted"] + metrics["submitted"] + metrics["discarded_on_abort"]
        )
        assert metrics["submitted"] == metrics["completed"] + metrics["failed_in_flight"]
        assert metrics["captured"] == (
            metrics["completed"] + metrics["evicted"] + metrics["failed"] + metrics["aborted"]
        )
        if variable in ("LESSON21_FAIL_POSTPROCESS_BATCH", "LESSON21_ABORT_AFTER_SUBMISSIONS"):
            assert metrics["failed_in_flight"] > 0
        print(f"{variable}: intended failure and terminal accounting verified")


if __name__ == "__main__":
    main()
