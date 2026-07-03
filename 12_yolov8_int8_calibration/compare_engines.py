#!/usr/bin/env python3
"""Compare FP32, FP16, and INT8 YOLOv8 TensorRT engines on a small validation set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from cuda.bindings import runtime as cudart

LESSON09 = Path(__file__).resolve().parents[1] / "09_yolov8_trt_python"
sys.path.insert(0, str(LESSON09))

import infer_yolov8_trt as yolo_ref  # noqa: E402


def image_paths(directory: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = sorted(path for path in directory.rglob("*") if path.suffix.lower() in suffixes)
    if not paths:
        raise FileNotFoundError(f"no validation images found under {directory}")
    return paths


def run_engine(engine_path: Path,
               image_path: Path,
               confidence: float,
               iou: float,
               max_detections: int) -> dict[str, Any]:
    yolo_ref.check_cuda(cudart.cudaSetDevice(0), "cudaSetDevice")
    image = yolo_ref.read_image(image_path)
    engine = yolo_ref.load_engine(engine_path)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(f"failed to create context for {engine_path}")

    input_names = [name for name in yolo_ref.tensor_names(engine) if yolo_ref.tensor_mode(engine, name) == "input"]
    if len(input_names) != 1:
        raise RuntimeError(f"expected one input tensor, got {input_names}")
    input_name = input_names[0]
    input_shape = yolo_ref.tensor_shape(engine, context, input_name)
    t0 = time.perf_counter()
    input_tensor, letterbox_info = yolo_ref.preprocess(image, input_shape)
    bindings, pointer_table, allocations = yolo_ref.allocate_bindings(engine, context)
    try:
        outputs = yolo_ref.execute(context, bindings, pointer_table, input_tensor)
    finally:
        for allocation in allocations:
            allocation.close()
    t1 = time.perf_counter()
    output_name, output_tensor = next(iter(outputs.items()))
    detections = yolo_ref.decode_yolov8(output_tensor, letterbox_info, confidence, iou, max_detections)
    return {
        "engine": str(engine_path),
        "image": str(image_path),
        "input_shape": list(input_shape),
        "output_name": output_name,
        "output_shape": list(output_tensor.shape),
        "output": output_tensor,
        "detections": [asdict(det) for det in detections],
        "latency_ms": (t1 - t0) * 1000.0,
    }


def tensor_drift(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} vs {candidate.shape}")
    abs_error = np.abs(reference.astype(np.float32) - candidate.astype(np.float32))
    return {
        "max_abs": float(np.max(abs_error)),
        "mean_abs": float(np.mean(abs_error)),
        "p99_abs": float(np.percentile(abs_error, 99)),
    }


def top_detection_signature(result: dict[str, Any]) -> dict[str, Any] | None:
    detections = result["detections"]
    if not detections:
        return None
    top = max(detections, key=lambda det: det["confidence"])
    return {
        "class_id": top["class_id"],
        "class_name": top["class_name"],
        "confidence": top["confidence"],
        "box_xyxy": top["box_xyxy"],
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    validation_images = image_paths(args.validation_dir)
    engines = {
        "fp32": args.fp32_engine,
        "fp16": args.fp16_engine,
        "int8": args.int8_engine,
    }

    results: list[dict[str, Any]] = []
    for image in validation_images:
        per_engine = {
            name: run_engine(path, image, args.confidence, args.iou, args.max_detections)
            for name, path in engines.items()
            if path is not None and path.is_file()
        }
        if "fp32" not in per_engine:
            raise FileNotFoundError("FP32 engine is required for drift reference")

        comparisons: dict[str, Any] = {}
        fp32_output = per_engine["fp32"]["output"]
        fp32_top = top_detection_signature(per_engine["fp32"])
        for name, result in per_engine.items():
            if name == "fp32":
                continue
            candidate_top = top_detection_signature(result)
            comparisons[name] = {
                "tensor_drift_vs_fp32": tensor_drift(fp32_output, result["output"]),
                "detection_count_delta": len(result["detections"]) - len(per_engine["fp32"]["detections"]),
                "top_detection_changed": (fp32_top or {}).get("class_id") != (candidate_top or {}).get("class_id"),
                "top_detection": candidate_top,
            }

        serializable_engines = {}
        for name, result in per_engine.items():
            serializable = dict(result)
            serializable.pop("output")
            serializable_engines[name] = serializable

        results.append(
            {
                "image": str(image),
                "engines": serializable_engines,
                "fp32_top_detection": fp32_top,
                "comparisons": comparisons,
            }
        )

    return {
        "validation_dir": str(args.validation_dir),
        "engine_paths": {name: str(path) for name, path in engines.items() if path is not None},
        "images": results,
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Lesson 12 INT8 Comparison Report",
        "",
        f"- Validation directory: `{report['validation_dir']}`",
        "",
        "## Engine Paths",
        "",
    ]
    for name, engine_path in report["engine_paths"].items():
        lines.append(f"- {name}: `{engine_path}`")
    lines.extend(["", "## Image Results", ""])
    for item in report["images"]:
        lines.append(f"### `{item['image']}`")
        top = item["fp32_top_detection"]
        lines.append(f"- FP32 top detection: `{top}`")
        for name, comparison in item["comparisons"].items():
            drift = comparison["tensor_drift_vs_fp32"]
            lines.append(
                f"- {name}: max_abs={drift['max_abs']:.6f}, mean_abs={drift['mean_abs']:.6f}, "
                f"p99_abs={drift['p99_abs']:.6f}, count_delta={comparison['detection_count_delta']}, "
                f"top_changed={comparison['top_detection_changed']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", type=Path, default=Path("data/validation_smoke"))
    parser.add_argument("--fp32-engine", type=Path, default=Path("../06_trtexec_engine/outputs/yolov8n_static_fp32.engine"))
    parser.add_argument("--fp16-engine", type=Path, default=Path("../06_trtexec_engine/outputs/yolov8n_static_fp16.engine"))
    parser.add_argument("--int8-engine", type=Path, default=Path("outputs/yolov8n_static_int8.engine"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-detections", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = compare(args)
    json_path = args.output_dir / "int8_comparison_report.json"
    md_path = args.output_dir / "int8_comparison_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
