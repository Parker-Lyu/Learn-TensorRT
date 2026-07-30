#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def render(sources: list[Path]) -> str:
    if len(sources) < 2:
        raise ValueError("at least two video sources are required")
    sections = []
    for index, source in enumerate(sources):
        absolute = source.expanduser().resolve()
        if not absolute.is_file():
            raise FileNotFoundError(f"missing video source: {absolute}")
        sections.append(f"[source{index}]\nenable=1\ntype=3\nuri=file://{absolute}\nnum-sources=1\n")
    count = len(sources)
    return """[application]
enable-perf-measurement=1
perf-measurement-interval-sec=5

[tiled-display]
enable=1
rows=1
columns={count}
width=1280
height=720

[streammux]
gpu-id=0
batch-size={count}
batched-push-timeout=40000
width=640
height=640
live-source=0
enable-padding=1

[primary-gie]
enable=1
config-file=../config/config_infer_primary_yolov8.txt

[osd]
enable=0

[sink0]
enable=1
type=1
sync=0
qos=0

""".format(count=count) + "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/deepstream_app_config.txt")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args.source), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
