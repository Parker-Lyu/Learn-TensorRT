#!/usr/bin/env python3
import json,subprocess,sys,tempfile
from pathlib import Path
exe,engine,image=sys.argv[1:]
with tempfile.TemporaryDirectory() as root:
 p=Path(root)
 for batch in (1,4):
  subprocess.run([exe,engine,image,"4",str(batch),"2",str(p/str(batch))],check=True,capture_output=True,text=True)
 one=[json.loads(x) for x in (p/"1/detections.jsonl").read_text().splitlines()]
 four=[json.loads(x) for x in (p/"4/detections.jsonl").read_text().splitlines()]
 assert len(one)==len(four)==4
 for a,b in zip(one,four):
  assert len(a["detections"])==len(b["detections"])
  for da,db in zip(a["detections"],b["detections"]):
   assert da["class_id"]==db["class_id"]
   assert abs(da["confidence"]-db["confidence"])<1e-4
   assert max(abs(x-y) for x,y in zip(da["box"],db["box"]))<0.1
