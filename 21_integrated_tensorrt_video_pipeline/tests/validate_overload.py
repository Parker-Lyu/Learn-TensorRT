#!/usr/bin/env python3
import json,sys
from pathlib import Path
m=json.loads((Path(sys.argv[1])/"metrics.json").read_text())
assert m["overload_policy"]=="drop-oldest"
assert m["queue_peak"]<=1
assert m["dropped"]>0
assert m["captured"]==m["processed"]+m["dropped"]
# One slot may grow once for this fixed image/batch shape; steady-state submissions reuse it.
assert sum(sample["capacity_growth_ms"]>0 for sample in m["batch_samples"])<=m["slots"]
