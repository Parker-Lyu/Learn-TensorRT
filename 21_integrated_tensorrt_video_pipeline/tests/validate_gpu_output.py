#!/usr/bin/env python3
import json, sys
from pathlib import Path
records=[json.loads(line) for line in (Path(sys.argv[1])/"detections.jsonl").read_text().splitlines()]
metrics=json.loads((Path(sys.argv[1])/"metrics.json").read_text())
assert len(records)==8
assert {r["stream_id"] for r in records}=={0,1}
assert {(r["stream_id"],r["frame_id"]) for r in records}=={(stream,frame) for stream in (0,1) for frame in range(4)}
assert len({(r["stream_id"],r["frame_id"]) for r in records})==8
assert metrics["captured"]==metrics["processed"]==8
assert metrics["slots"]==2 and metrics["batches"]==2
assert metrics["preprocess_ms"]>0 and metrics["inference_ms"]>0
assert metrics["host_staging_ms"]>0 and metrics["h2d_ms"]>0 and metrics["d2h_ms"]>0
assert metrics["clock_domains"]["host"]=="std::chrono::steady_clock"
assert "CUDA events" in metrics["clock_domains"]["gpu"]
assert all(sample["h2d_ms"]>0 and sample["tensorrt_ms"]>0 and sample["d2h_ms"]>0
           for sample in metrics["batch_samples"])

assert metrics["batch_distribution"]=={"4":2}
assert metrics["per_stream_processed"]=={"0":4,"1":4}
assert metrics["environment"]["tensorrt"].startswith("10.14")
assert metrics["environment"]["cuda_runtime"] >= 13000
assert metrics["environment"]["gpu"]
