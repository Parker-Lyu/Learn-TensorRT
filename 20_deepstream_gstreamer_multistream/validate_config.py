#!/usr/bin/env python3
from __future__ import annotations

import configparser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def validate_infer(path: Path) -> dict:
    parser = configparser.ConfigParser()
    parser.read(path)
    if "property" not in parser:
        raise ValueError("missing [property] section")
    props = parser["property"]
    required = {"model-engine-file", "batch-size", "num-detected-classes",
                "parse-bbox-func-name", "custom-lib-path", "output-blob-names"}
    missing = required - props.keys()
    if missing:
        raise ValueError(f"missing inference properties: {sorted(missing)}")
    if props.getint("batch-size") < 2 or props.getint("num-detected-classes") != 80:
        raise ValueError("lesson 20 requires batch >=2 and 80 COCO classes")
    if props["parse-bbox-func-name"] != "NvDsInferParseYoloV8":
        raise ValueError("unexpected parser function")
    labels = path.parent / props["labelfile-path"]
    if len(labels.read_text(encoding="utf-8").splitlines()) != 80:
        raise ValueError("COCO label file must contain 80 classes")
    return {"batch_size": props.getint("batch-size"), "classes": 80}


if __name__ == "__main__":
    print(validate_infer(ROOT / "config/config_infer_primary_yolov8.txt"))
