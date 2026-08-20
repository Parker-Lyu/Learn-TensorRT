#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def render(sources: list[Path], inference_config: str = "../config/config_infer_primary_yolov8.txt",
           render_output: Path | None = None) -> str:
    if len(sources) < 2:
        raise ValueError("at least two video sources are required")
    sections = []
    for index, source in enumerate(sources):
        absolute = source.expanduser().resolve()
        if not absolute.is_file():
            raise FileNotFoundError(f"missing video source: {absolute}")
        sections.append(f"[source{index}]\nenable=1\ntype=3\nuri=file://{absolute}\nnum-sources=1\n")
    count = len(sources)
    if render_output is None:
        osd_section = """[osd]
enable=0

[sink0]
enable=1
type=1
sync=0
qos=0
"""
    else:
        osd_section = """[osd]
enable=1

[sink0]
enable=1
type=3
sync=0
qos=0
container=1
codec=1
bitrate=4000000
iframeinterval=30
output-file={render_output}
""".format(render_output=render_output)

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
config-file={inference_config}

{osd_section}
""".format(count=count, inference_config=inference_config, osd_section=osd_section) + "\n".join(sections)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--inference-config", default="../config/config_infer_primary_yolov8.txt",
                        help="path to the nvinfer config, relative to the generated app config")
    parser.add_argument("--render-output", type=Path,
                        help="enable OSD and save an annotated MP4 to this path")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/deepstream_app_config.txt")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_output = args.render_output.expanduser().resolve() if args.render_output else None
    if render_output:
        render_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args.source, args.inference_config, render_output), encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
