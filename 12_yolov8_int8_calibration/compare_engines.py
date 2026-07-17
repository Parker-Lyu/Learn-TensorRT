#!/usr/bin/env python3
"""Evaluate PyTorch and TensorRT FP32/FP16/INT8 on one fixed labeled split."""

from __future__ import annotations

import argparse
import copy
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
from evaluation import (
    allocate_prediction_buffer,
    append_predictions,
    detection_metrics_packed,
    load_ground_truth,
    tensor_drift,
)

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
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        tensor = self.torch.from_numpy(input_tensor).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        output_array = output.detach().float().cpu().numpy()
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
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
    expected_device = f"cuda:{args.gpu}"
    normalized_device = "cuda:0" if args.device.lower() == "cuda" else args.device.lower()
    if normalized_device != expected_device:
        raise ValueError(
            f"--device must be {expected_device} when --gpu is {args.gpu}; "
            "all backends must run on the same GPU"
        )
    for name in ("max_map50_95_drop", "max_map50_drop", "max_precision_drop", "max_recall_drop"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative")


def current_software() -> dict[str, str]:
    return {
        "tensorrt": trt.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": importlib.metadata.version("torch"),
        "ultralytics": importlib.metadata.version("ultralytics"),
    }


def regression_thresholds(args: argparse.Namespace) -> dict[str, float]:
    return {
        "map50_95": args.max_map50_95_drop,
        "map50": args.max_map50_drop,
        "precision": args.max_precision_drop,
        "recall": args.max_recall_drop,
    }


def expected_settings(args: argparse.Namespace, input_shape: tuple[int, ...]) -> dict[str, Any]:
    return {
        "confidence": args.confidence,
        "nms_iou": args.iou,
        "max_detections": args.max_detections,
        "input_shape": list(input_shape),
        "warmup": args.warmup,
        "metric_implementation": "course-coco-like-101point-v2-no-crowd-no-area-ranges",
    }


def reference_artifact_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "pytorch_weights": args.weights,
        "tensorrt_fp32": args.fp32_engine,
        "tensorrt_fp16": args.fp16_engine,
    }


def load_validated_reference_report(
    args: argparse.Namespace, manifest: dict[str, Any], input_shape: tuple[int, ...]
) -> dict[str, Any]:
    if args.reference_report is None:
        raise ValueError("reference report path is required")
    report = json.loads(args.reference_report.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1:
        raise ValueError("reference report has an unsupported schema")

    expected_manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    dataset = report.get("dataset", {})
    if dataset.get("manifest_sha256") != expected_manifest_hash:
        raise ValueError("reference report manifest hash does not match the requested manifest")
    if dataset.get("dataset_id") != manifest["dataset_id"]:
        raise ValueError("reference report dataset ID does not match the requested manifest")

    settings = report.get("settings", {})
    for name, expected in expected_settings(args, input_shape).items():
        if settings.get(name) != expected:
            raise ValueError(
                f"reference report setting {name!r} changed: "
                f"expected {expected!r}, found {settings.get(name)!r}"
            )

    expected_thresholds = {
        f"max_{name}_drop": value for name, value in regression_thresholds(args).items()
    }
    if report.get("regression_thresholds") != expected_thresholds:
        raise ValueError("reference report regression thresholds do not match this evaluation")

    artifacts = report.get("artifacts", {})
    for name, path in reference_artifact_paths(args).items():
        declared = artifacts.get(name, {})
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if declared.get("sha256") != actual_hash:
            raise ValueError(f"reference artifact hash changed: {name}")

    if report.get("software") != current_software():
        raise ValueError("reference report software identity does not match the current environment")
    for name in ("pytorch", "tensorrt_fp32", "tensorrt_fp16"):
        backend = report.get("backends", {}).get(name)
        if not isinstance(backend, dict) or not backend.get("passed"):
            raise ValueError(f"reference report has no passing {name} backend")
    return report


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

        ground_truth: dict[int, list[dict]] = {}
        prediction_buffers = {
            name: allocate_prediction_buffer(len(records), args.max_detections)
            for name in runners
        }
        prediction_offsets = {name: 0 for name in runners}
        latencies = {name: [] for name in runners}
        drift_maxima = {
            name: {key: 0.0 for key in ("max_abs", "mean_abs", "p99_abs")}
            for name in trt_runners
        }
        changed_examples: list[dict] = []
        for image_index, record in enumerate(records):
            image_path = resolve_path(args.manifest, record["image"])
            label_path = resolve_path(args.manifest, record["label"])
            ground_truth[image_index] = load_ground_truth(image_path, label_path)
            per_backend = {
                name: runner.run(image_path, args.confidence, args.iou, args.max_detections)
                for name, runner in runners.items()
            }
            fp32_output = per_backend["tensorrt_fp32"]["output"]
            fp32_detections = per_backend["tensorrt_fp32"]["detections"]
            example = {"image": record["image"], "comparisons": {}}
            example_changed = False
            for name, result in per_backend.items():
                prediction_offsets[name] = append_predictions(
                    prediction_buffers[name],
                    prediction_offsets[name],
                    image_index,
                    result["detections"],
                )
                latencies[name].append(result["latency_ms"])
                if name in trt_runners:
                    drift = tensor_drift(fp32_output, result["output"])
                    for key, value in drift.items():
                        drift_maxima[name][key] = max(drift_maxima[name][key], value)
                    changed = changed_detection(fp32_detections, result["detections"])
                    example["comparisons"][name] = {
                        "tensor_drift_vs_fp32": drift,
                        "detection_count": len(result["detections"]),
                        "changed_vs_fp32": changed,
                    }
                    example_changed = (
                        example_changed or changed or drift["p99_abs"] >= args.inspect_p99
                    )
            if example_changed and len(changed_examples) < args.max_inspection_examples:
                changed_examples.append(example)
            if (image_index + 1) % 100 == 0 or image_index + 1 == len(records):
                print(f"Evaluated images: {image_index + 1}/{len(records)}", flush=True)
    finally:
        for runner in trt_runners.values():
            runner.close()

    metrics = {}
    for name in prediction_buffers:
        print(f"Computing detection metrics: {name}", flush=True)
        metrics[name] = detection_metrics_packed(
            prediction_buffers[name][:prediction_offsets[name]], ground_truth
        )
    reference = metrics["pytorch"]
    thresholds = regression_thresholds(args)
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
        if name in drift_maxima:
            backends[name]["tensor_drift_vs_fp32"] = drift_maxima[name]

    return {
        "schema_version": 1,
        "dataset": {
            "manifest": str(args.manifest),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "dataset_id": manifest["dataset_id"],
            "validation_images": len(records),
        },
        "settings": {
            "confidence": args.confidence,
            "nms_iou": args.iou,
            "max_detections": args.max_detections,
            "input_shape": list(input_shape),
            "warmup": args.warmup,
            "metric_implementation": "course-coco-like-101point-v2-no-crowd-no-area-ranges",
            "latency_scope": (
                "runtime wrapper with H2D, inference, D2H; excludes image loading, "
                "preprocessing, and decode"
            ),
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
        "software": current_software(),
        "regression_thresholds": {f"max_{name}_drop": value for name, value in thresholds.items()},
        "backends": backends,
        "changed_or_high_drift_examples": changed_examples[: args.max_inspection_examples],
        "release_gate": {"passed": not failures, "failed_backends": failures},
    }


def evaluate_candidate_with_references(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    manifest = load_manifest(args.manifest)
    records = [record for record in manifest["records"] if record["split"] == "validation"]
    if not records:
        raise ValueError("manifest contains no validation records")

    yolo_ref.check_cuda(cudart.cudaSetDevice(args.gpu), "cudaSetDevice")
    runner = TensorRtRunner(args.int8_engine)
    try:
        input_shape = runner.input_shape
        reference = load_validated_reference_report(args, manifest, input_shape)
        first_image = resolve_path(args.manifest, records[0]["image"])
        for _ in range(args.warmup):
            runner.run(first_image, args.confidence, args.iou, args.max_detections)

        ground_truth: dict[int, list[dict]] = {}
        predictions = allocate_prediction_buffer(len(records), args.max_detections)
        prediction_offset = 0
        latencies = []
        for image_index, record in enumerate(records):
            image_path = resolve_path(args.manifest, record["image"])
            label_path = resolve_path(args.manifest, record["label"])
            ground_truth[image_index] = load_ground_truth(image_path, label_path)
            result = runner.run(image_path, args.confidence, args.iou, args.max_detections)
            prediction_offset = append_predictions(
                predictions, prediction_offset, image_index, result["detections"]
            )
            latencies.append(result["latency_ms"])
            if (image_index + 1) % 100 == 0 or image_index + 1 == len(records):
                print(f"Evaluated candidate images: {image_index + 1}/{len(records)}", flush=True)
    finally:
        runner.close()

    print("Computing detection metrics: tensorrt_int8", flush=True)
    metrics = detection_metrics_packed(predictions[:prediction_offset], ground_truth)
    thresholds = regression_thresholds(args)
    pytorch_metrics = reference["backends"]["pytorch"]["metrics"]
    deltas = {name: metrics[name] - pytorch_metrics[name] for name in thresholds}
    passed = all(deltas[name] >= -allowed for name, allowed in thresholds.items())
    candidate_backend = {
        "metrics": metrics,
        "delta_vs_pytorch": deltas,
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
        },
        "passed": passed,
        "diagnostics": (
            "candidate-only mode does not recompute FP32 tensor drift or changed examples"
        ),
    }
    backends = {
        name: copy.deepcopy(reference["backends"][name])
        for name in ("pytorch", "tensorrt_fp32", "tensorrt_fp16")
    }
    backends["tensorrt_int8"] = candidate_backend
    artifacts = {
        name: copy.deepcopy(reference["artifacts"][name])
        for name in ("pytorch_weights", "tensorrt_fp32", "tensorrt_fp16")
    }
    artifacts["tensorrt_int8"] = {
        "path": str(args.int8_engine),
        "sha256": hashlib.sha256(args.int8_engine.read_bytes()).hexdigest(),
    }
    return {
        "schema_version": 1,
        "evaluation_mode": "candidate_only_with_reused_references",
        "reference_report": {
            "path": str(args.reference_report),
            "sha256": hashlib.sha256(args.reference_report.read_bytes()).hexdigest(),
        },
        "dataset": {
            "manifest": str(args.manifest),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "dataset_id": manifest["dataset_id"],
            "validation_images": len(records),
        },
        "settings": {
            **expected_settings(args, input_shape),
            "latency_scope": (
                "runtime wrapper with H2D, inference, D2H; excludes image loading, "
                "preprocessing, and decode"
            ),
        },
        "artifacts": artifacts,
        "software": current_software(),
        "regression_thresholds": {
            f"max_{name}_drop": value for name, value in thresholds.items()
        },
        "backends": backends,
        "changed_or_high_drift_examples": [],
        "release_gate": {
            "passed": passed,
            "failed_backends": [] if passed else ["tensorrt_int8"],
        },
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
    source_note = (
        "The JSON file is the source of truth for thresholds, deltas, reused-reference identity, "
        "and candidate metrics. Candidate-only mode does not collect FP32 tensor drift."
        if report.get("evaluation_mode") == "candidate_only_with_reused_references"
        else (
            "The JSON file is the source of truth for thresholds, deltas, tensor drift, "
            "and inspection examples."
        )
    )
    lines.extend([
        "",
        f"Release gate: **{'PASS' if gate['passed'] else 'FAIL'}**",
        "",
        source_note,
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
    parser.add_argument(
        "--reference-report",
        type=Path,
        help=(
            "Reuse validated PyTorch/FP32/FP16 metrics from a prior full report and run only "
            "the INT8 candidate."
        ),
    )
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
    if args.reference_report is not None and not args.reference_report.is_file():
        raise FileNotFoundError(f"reference report not found: {args.reference_report}")
    report = (
        evaluate_candidate_with_references(args)
        if args.reference_report is not None
        else evaluate(args)
    )
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
