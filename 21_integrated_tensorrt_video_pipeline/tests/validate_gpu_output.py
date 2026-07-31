#!/usr/bin/env python3
import json, sys
from pathlib import Path
records=[json.loads(line) for line in (Path(sys.argv[1])/"detections.jsonl").read_text().splitlines()]
metrics=json.loads((Path(sys.argv[1])/"metrics.json").read_text())
assert len(records)==8
assert {r["stream_id"] for r in records}=={0,1}
assert [r["frame_id"] for r in records]==list(range(8))
assert len({r["frame_id"] for r in records})==8
assert metrics["captured"]==metrics["processed"]==8
assert metrics["slots"]==2 and metrics["batches"]==2
assert metrics["preprocess_ms"]>0 and metrics["inference_ms"]>0
