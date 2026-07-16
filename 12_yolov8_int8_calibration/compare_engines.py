#!/usr/bin/env python3
"""Evaluate PyTorch and TensorRT FP32/FP16/INT8 on one fixed labeled split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import tensorrt as trt
from cuda.bindings import runtime as cudart

from dataset_manifest import DEFAULT_COCO_MANIFEST, load_manifest, resolve_path
from evaluation import detection_metrics, load_ground_truth, tensor_drift

REPO_ROOT = Path(__file__).resolve().parents[1]
LESSON09 = REPO_ROOT / "09_yolov8_trt_python"
sys.path.insert(0, str(LESSON09))
import infer_yolov8_trt as yolo_ref  # noqa: E402


class TensorRtRunner:
    def __init__(self, engine_path: Path) -> None:
        self.path = engine_path
        self.engine = yolo_ref.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"failed to create execution context: {engine_path}")
        input_names = [
            name for name in yolo_ref.tensor_names(self.engine)
            if yolo_ref.tensor_mode(self.engine, name) == "input"
        ]
        if len(input_names) != 1:
            raise RuntimeError(f"expected one input tensor in {engine_path}, got {input_names}")
        self.input_shape = yolo_ref.tensor_shape(self.engine, self.context, input_names[0])
        self.bindings, self.pointer_table, self.allocations = yolo_ref.allocate_bindings(
            self.engine, self.context
        )

    def run(self, image_path: Path, confidence: float, iou: float, max_detections: int) -> dict:
        image = yolo_ref.read_image(image_path)
        input_tensor, letterbox_info = yolo_ref.preprocess(image, self.input_shape)
        started = time.perf_counter()
        outputs = yolo_ref.execute(self.context, self.bindings, self.pointer_table, input_tensor)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        output_name, output = next(iter(outputs.items()))
        detections = yolo_ref.decode_yolov8(
            output, letterbox_info, confidence, iou, max_detections
        )
        return {
            "output_name": output_name,
            "output": output,
            "detections": [asdict(item) for item in detections],
            "latency_ms": elapsed_ms,
        }

    def close(self) -> None:
        for allocation in self.allocations:
            allocation.close()


class PyTorchRunner:
    def __init__(self, weights: Path, device: str, input_shape: tuple[int, ...]) -> None:
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.device = torch.device(device)
        self.model = YOLO(str(weights)).model.to(self.device).eval()
        self.input_shape = input_shape

    def run(self, image_path: Path, confidence: float, iou: float, max_detections: int) -> dict:
        image = yolo_ref.read_image(image_path)
        input_tensor, letterbox_info = yolo_ref.preprocess(image, self.input_shape)
        tensor = self.torch.from_numpy(input_tensor).to(self.device)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model(tensor)
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if isinstance(output, (tuple, list)):
            output = output[0]
        output_array = output.detach().float().cpu().numpy()
        detections = yolo_ref.decode_yolov8(
            output_array, letterbox_info, confidence, iou, max_detections
        )
        return {
            "output_name": "pytorch_output",
            "output": output_array,
            "detections": [asdict(item) for item in detections],
            "latency_ms": elapsed_ms,
        }


def create_trt_runners(engine_paths: dict[str, Path]) -> dict[str, TensorRtRunner]:
    runners: dict[str, TensorRtRunner] = {}
    try:
        for name, path in engine_paths.items():
            runners[name] = TensorRtRunner(path)
    except Exception:
        for runner in runners.values():
            runner.close()
        raise
    return runners


def validate_args(args: argparse.Namespace) -> None:
    required = [args.weights, args.fp32_engine, args.fp16_engine, args.int8_engine, args.manifest]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("required evaluation artifact(s) missing: " + ", ".join(missing))
    if not 0.0 <= args.confidence <= 1.0 or not 0.0 <= args.iou <= 1.0:
        raise ValueError("--confidence and --iou must be in [0, 1]")
    if args.max_detections <= 0 or args.warmup < 0:
        raise ValueError("--max-detections must be positive and --warmup cannot be negative")
    if args.gpu < 0 or args.inspect_p99 < 0.0 or args.max_inspection_examples <= 0:
        raise ValueError("GPU index/inspection threshold/count are outside their valid range")
    for name in ("max_map50_95_drop", "max_map50_drop", "max_precision_drop", "max_recall_drop"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")


def changed_detection(reference: list[dict], candidate: list[dict]) -> bool:
    return len(reference) != len(candidate) or [item["class_id"] for item in reference] != [
        item["class_id"] for item in candidate
    ]


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    manifest = load_manifest(args.manifest)
    records = [record for record in manifest["records"] if record["split"] == "validation"]
    if not records:
        raise ValueError("manifest contains no validation records")

    yolo_ref.check_cuda(cudart.cudaSetDevice(args.gpu), "cudaSetDevice")
    trt_runners = create_trt_runners({
        "tensorrt_fp32": args.fp32_engine,
        "tensorrt_fp16": args.fp16_engine,
        "tensorrt_int8": args.int8_engine,
    })
    try:
        input_shape = trt_runners["tensorrt_fp32"].input_shape
        for name, runner in trt_runners.items():
            if runner.input_shape != input_shape:
                raise ValueError(
                    f"{name} input shape {runner.input_shape} != FP32 {input_shape}"
                )
        runners: dict[str, Any] = {
            "pytorch": PyTorchRunner(args.weights, args.device, input_shape),
            **trt_runners,
        }
    except Exception:
        for runner in trt_runners.values():
            runner.close()
        raise

    try:
        first_image = resolve_path(args.manifest, records[0]["image"])
        for _ in range(args.warmup):
            for runner in runners.values():
                runner.run(first_image, args.confidence, args.iou, args.max_detections)

        ground_truth: dict[str, list[dict]] = {}
        predictions = {name: {} for name in runners}
        latencies = {name: [] for name in runners}
        drift_values = {name: [] for name in trt_runners}
        changed_examples: list[dict] = []
        for record in records:
            image_path = resolve_path(args.manifest, record["image"])
            label_path = resolve_path(args.manifest, record["label"])
            image_id = record["image_sha256"]
            ground_truth[image_id] = load_ground_truth(image_path, label_path)
            per_backend = {
                name: runner.run(image_path, args.confidence, args.iou, args.max_detections)
                for name, runner in runners.items()
            }
            fp32_output = per_backend["tensorrt_fp32"]["output"]
            fp32_detections = per_backend["tensorrt_fp32"]["detections"]
            example = {"image": record["image"], "comparisons": {}}
            example_changed = False
            for name, result in per_backend.items():
                predictions[name][image_id] = result["detections"]
                latencies[name].append(result["latency_ms"])
                if name in trt_runners:
                    drift = tensor_drift(fp32_output, result["output"])
                    drift_values[name].append(drift)
                    changed = changed_detection(fp32_detections, result["detections"])
                    example["comparisons"][name] = {
                        "tensor_drift_vs_fp32": drift,
                        "detection_count": len(result["detections"]),
                        "changed_vs_fp32": changed,
                    }
                    example_changed = (
                        example_changed or changed or drift["p99_abs"] >= args.inspect_p99
                    )
            if example_changed:
                changed_examples.append(example)
    finally:
        for runner in trt_runners.values():
            runner.close()

    metrics = {name: detection_metrics(items, ground_truth) for name, items in predictions.items()}
    reference = metrics["pytorch"]
    thresholds = {
        "map50_95": args.max_map50_95_drop,
        "map50": args.max_map50_drop,
        "precision": args.max_precision_drop,
        "recall": args.max_recall_drop,
    }
    backends = {}
    failures = []
    for name, backend_metrics in metrics.items():
        deltas = {metric: backend_metrics[metric] - reference[metric] for metric in thresholds}
        passed = all(deltas[metric] >= -allowed for metric, allowed in thresholds.items())
        if not passed:
            failures.append(name)
        backends[name] = {
            "metrics": backend_metrics,
            "delta_vs_pytorch": deltas,
            "latency_ms": {
                "mean": statistics.fmean(latencies[name]),
                "p50": float(np.percentile(latencies[name], 50)),
                "p90": float(np.percentile(latencies[name], 90)),
            },
            "passed": passed,
        }
        if name in drift_values:
            backends[name]["tensor_drift_vs_fp32"] = {
                key: float(np.max([item[key] for item in drift_values[name]]))
                for key in ("max_abs", "mean_abs", "p99_abs")
            }

    return {
        "schema_version": 1,
        "dataset": {
            "manifest": str(args.manifest),
            "dataset_id": manifest["dataset_id"],
            "validation_images": len(records),
        },
        "settings": {
            "confidence": args.confidence,
            "nms_iou": args.iou,
            "max_detections": args.max_detections,
            "input_shape": list(input_shape),
            "warmup": args.warmup,
        },
        "artifacts": {
            name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in {
                "pytorch_weights": args.weights,
                "tensorrt_fp32": args.fp32_engine,
                "tensorrt_fp16": args.fp16_engine,
                "tensorrt_int8": args.int8_engine,
            }.items()
        },
        "software": {
            "tensorrt": trt.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "torch": importlib.metadata.version("torch"),
            "ultralytics": importlib.metadata.version("ultralytics"),
        },
        "regression_thresholds": {f"max_{name}_drop": value for name, value in thresholds.items()},
        "backends": backends,
        "changed_or_high_drift_examples": changed_examples[: args.max_inspection_examples],
        "release_gate": {"passed": not failures, "failed_backends": failures},
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Lesson 12 Precision Evaluation",
        "",
        f"Dataset: `{report['dataset']['dataset_id']}` "
        f"({report['dataset']['validation_images']} images)",
        "",
        "| Backend | mAP50-95 | mAP50 | Precision | Recall | Mean latency (ms) | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, result in report["backends"].items():
        metric = result["metrics"]
        lines.append(
            f"| {name} | {metric['map50_95']:.4f} | {metric['map50']:.4f} | "
            f"{metric['precision']:.4f} | {metric['recall']:.4f} | "
            f"{result['latency_ms']['mean']:.3f} | {'PASS' if result['passed'] else 'FAIL'} |"
        )
    gate = report["release_gate"]
    lines.extend([
        "",
        f"Release gate: **{'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        "The JSON file is the source of truth for thresholds, deltas, tensor drift, "
        "and inspection examples.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_COCO_MANIFEST)
    parser.add_argument("--weights", type=Path, default=Path("../assets/yolov8n.pt"))
    parser.add_argument(
        "--fp32-engine", type=Path,
        default=Path("../06_trtexec_engine/outputs/yolov8n_static_fp32.engine")
    )
    parser.add_argument(
        "--fp16-engine", type=Path,
        default=Path("../06_trtexec_engine/outputs/yolov8n_static_fp16.engine")
    )
    parser.add_argument(
        "--int8-engine", type=Path,
        default=Path("outputs/yolov8n_static_int8.engine")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-map50-95-drop", type=float, default=0.02)
    parser.add_argument("--max-map50-drop", type=float, default=0.02)
    parser.add_argument("--max-precision-drop", type=float, default=0.03)
    parser.add_argument("--max-recall-drop", type=float, default=0.03)
    parser.add_argument("--inspect-p99", type=float, default=0.1)
    parser.add_argument("--max-inspection-examples", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.manifest.is_file():
        raise FileNotFoundError(
            f"COCO dataset manifest not found: {args.manifest}. Run "
            "`python3 assets/coco/prepare_coco.py` from the repository root."
        )
    report = evaluate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "precision_evaluation.json"
    markdown_path = args.output_dir / "precision_evaluation.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(markdown_path, report)
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {markdown_path}")
    print(f"Release gate: {'PASS' if report['release_gate']['passed'] else 'FAIL'}")
    return 0 if report["release_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
