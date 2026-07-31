#!/usr/bin/env python3
"""Copy a compatible generated TensorRT plan into the ignored Triton version directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_config(text: str) -> None:
    required = ['name: "yolov8"', 'platform: "tensorrt_plan"', "max_batch_size: 4",
                'name: "images"', 'name: "output0"', "dynamic_batching"]
    missing = [token for token in required if token not in text]
    if missing:
        raise ValueError(f"Triton config is missing: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", type=Path,
                        default=ROOT / "17_dynamic_batching/outputs/yolov8n_batch1_4_fp16.engine")
    args = parser.parse_args()
    engine = args.engine
    if not engine.is_file():
        raise FileNotFoundError("build lesson 17's dynamic engine first")
    config = ROOT / "24_triton_inference_server/model_repository/yolov8/config.pbtxt"
    validate_config(config.read_text(encoding="utf-8"))
    destination = config.parent / "1/model.plan"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(engine, destination)
    print(f"copied {engine.relative_to(ROOT)} -> {destination.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
