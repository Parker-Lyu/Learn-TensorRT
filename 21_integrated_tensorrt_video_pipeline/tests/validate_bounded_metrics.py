#!/usr/bin/env python3
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
raw_samples = (output / "batch_timing_samples.jsonl").read_text(encoding="utf-8").splitlines()

assert metrics["batches"] > metrics["batch_sample_storage"]["retained_in_metrics"]
assert len(metrics["batch_samples"]) <= metrics["batch_sample_storage"]["retained_in_metrics"]
assert len(raw_samples) == metrics["batches"]
assert all(json.loads(line)["capacity_growth_ms"] == 0.0 for line in raw_samples[2:])
