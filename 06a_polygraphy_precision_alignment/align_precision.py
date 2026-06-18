#!/usr/bin/env python3
"""Run Polygraphy precision checks and summarize ONNX Runtime versus TensorRT drift."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from polygraphy.comparator import RunResults
from polygraphy.json import load_json


REPO_ROOT = Path(__file__).resolve().parents[1]
LESSON_DIR = REPO_ROOT / "06a_polygraphy_precision_alignment"
DEFAULT_ONNX = REPO_ROOT / "05_torch_to_onnx" / "outputs" / "yolov8n.onnx"
DEFAULT_ENGINE = REPO_ROOT / "06_trtexec_engine" / "outputs" / "yolov8n_static_fp32.engine"
DEFAULT_INPUTS = LESSON_DIR / "outputs" / "input_data.json"
DEFAULT_OUTPUT_DIR = LESSON_DIR / "outputs"
POLYGRAPHY_CLI = LESSON_DIR / "polygraphy_cli_compat.py"
REPORT_SCOPE = "single_input_tensor_alignment"
REPORT_SCOPE_NOTE = (
    "This report compares raw model outputs for one controlled input tensor. It is useful "
    "precision-alignment evidence, but it is not dataset-level detection accuracy validation."
)


@dataclass(frozen=True)
class CommandRecord:
    name: str
    command: list[str]
    log_path: Path
    return_code: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Polygraphy to compare YOLO ONNX Runtime and TensorRT outputs."
    )
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="Validated ONNX model from lesson 05.")
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE, help="Serialized TensorRT engine from lesson 06.")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS, help="Polygraphy input JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Artifact directory.")
    parser.add_argument("--input-name", default="images", help="Input name for the TRT build fallback path.")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for report-level allclose.")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for report-level allclose.")
    parser.add_argument(
        "--polygraphy-rtol",
        type=float,
        default=None,
        help="Polygraphy CLI relative tolerance. Defaults to --rtol.",
    )
    parser.add_argument(
        "--polygraphy-atol",
        type=float,
        default=None,
        help="Polygraphy CLI absolute tolerance. Defaults to --atol.",
    )
    parser.add_argument(
        "--trt-mode",
        choices=("engine", "build"),
        default="engine",
        help="Use a serialized engine, or let Polygraphy build from ONNX.",
    )
    parser.add_argument(
        "--skip-trt",
        action="store_true",
        help="Only run ONNX Runtime. Useful for a quick smoke test without a GPU or engine.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Write summaries even if Polygraphy reports a tolerance failure.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    try:
        import polygraphy  # noqa: F401
    except Exception as exc:
        raise FileNotFoundError(
            "Polygraphy is not importable. Complete lesson 00 inside the TensorRT container."
        ) from exc
    if args.rtol < 0 or args.atol < 0:
        raise ValueError("--rtol and --atol must be non-negative")
    if args.polygraphy_rtol is not None and args.polygraphy_rtol < 0:
        raise ValueError("--polygraphy-rtol must be non-negative")
    if args.polygraphy_atol is not None and args.polygraphy_atol < 0:
        raise ValueError("--polygraphy-atol must be non-negative")
    if not args.onnx.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {args.onnx}\n"
            "Run lesson 05 first: python3 05_torch_to_onnx/export_yolov8_onnx.py"
        )
    if not args.inputs.exists():
        raise FileNotFoundError(
            f"Polygraphy input JSON not found: {args.inputs}\n"
            "Create it with: python3 06a_polygraphy_precision_alignment/make_polygraphy_inputs.py"
        )
    if args.trt_mode == "engine" and not args.skip_trt and not args.engine.exists():
        raise FileNotFoundError(
            f"TensorRT engine not found: {args.engine}\n"
            "Run lesson 06 first or use --trt-mode build to let Polygraphy build from ONNX."
        )
    if args.trt_mode == "build" and not args.input_name:
        raise ValueError("--input-name cannot be empty when --trt-mode build is used")


def command_preview(command: list[str]) -> str:
    if len(command) >= 2 and Path(command[1]).name == "polygraphy_cli_compat.py":
        return "polygraphy " + " ".join(command[2:])
    return " ".join(command)


def run_command(name: str, command: list[str], log_path: Path, keep_going: bool) -> CommandRecord:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n== {name} ==")
    print(command_preview(command))

    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(f"log: {log_path}")
    if completed.returncode != 0 and completed.stdout:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        print(tail)

    if completed.returncode != 0 and not keep_going:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}. See {log_path}")
    return CommandRecord(name=name, command=command, log_path=log_path, return_code=completed.returncode)


def polygraphy_command(*args: str | Path) -> list[str]:
    return [sys.executable, str(POLYGRAPHY_CLI), *[str(arg) for arg in args]]


def inspect_command(model_path: Path, model_type: str, log_path: Path) -> list[str]:
    return polygraphy_command(
        "inspect",
        "model",
        model_path,
        "--model-type",
        model_type,
        "--show",
        "layers",
        "--log-file",
        log_path,
        "--log-format",
        "no-colors",
    )


def onnxrt_command(args: argparse.Namespace, output_path: Path, log_path: Path) -> list[str]:
    return polygraphy_command(
        "run",
        args.onnx,
        "--onnxrt",
        "--load-inputs",
        args.inputs,
        "--save-outputs",
        output_path,
        "--log-file",
        log_path,
        "--log-format",
        "no-colors",
    )


def compare_command(args: argparse.Namespace, output_path: Path, log_path: Path) -> list[str]:
    rtol = args.polygraphy_rtol if args.polygraphy_rtol is not None else args.rtol
    atol = args.polygraphy_atol if args.polygraphy_atol is not None else args.atol

    if args.trt_mode == "engine":
        command = polygraphy_command(
            "run",
            args.engine,
            "--model-type",
            "engine",
            "--trt",
        )
    else:
        command = polygraphy_command(
            "run",
            args.onnx,
            "--onnxrt",
            "--trt",
            "--trt-min-shapes",
            f"{args.input_name}:[1,3,640,640]",
            "--trt-opt-shapes",
            f"{args.input_name}:[1,3,640,640]",
            "--trt-max-shapes",
            f"{args.input_name}:[1,3,640,640]",
        )

    if args.trt_mode == "engine":
        command.extend(["--load-outputs", str(output_path.parent / "onnxrt_outputs.json")])

    command.extend(
        [
            "--load-inputs",
            str(args.inputs),
            "--save-outputs",
            str(output_path),
            "--rtol",
            str(rtol),
            "--atol",
            str(atol),
            "--log-file",
            str(log_path),
            "--log-format",
            "no-colors",
        ]
    )
    return command


def load_runner_outputs(path: Path) -> dict[str, list[dict[str, np.ndarray]]]:
    raw = load_json(path)
    if isinstance(raw, RunResults):
        results = raw
    else:
        results = RunResults()
        results.load(raw)
    return {
        runner_name: [
            {output_name: np.asarray(output_value) for output_name, output_value in iteration.items()}
            for iteration in iterations
        ]
        for runner_name, iterations in results.items()
    }


def first_output(outputs: dict[str, list[dict[str, np.ndarray]]], preferred_runner: str | None = None) -> tuple[str, str, np.ndarray]:
    runner_names = list(outputs)
    if preferred_runner is not None:
        matches = [name for name in runner_names if preferred_runner.lower() in name.lower()]
        if matches:
            runner_names = matches
    if not runner_names:
        raise ValueError("no runner outputs found")
    runner_name = runner_names[0]
    iterations = outputs[runner_name]
    if not iterations:
        raise ValueError(f"runner {runner_name} has no iterations")
    output_map = iterations[0]
    if not output_map:
        raise ValueError(f"runner {runner_name} produced no outputs")
    output_name = next(iter(output_map))
    return runner_name, output_name, np.asarray(output_map[output_name])


def compare_arrays(reference: np.ndarray, candidate: np.ndarray, rtol: float, atol: float) -> dict[str, Any]:
    if reference.shape != candidate.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
            "allclose": False,
        }

    diff = np.abs(reference - candidate)
    max_index = np.unravel_index(int(np.argmax(diff)), diff.shape)
    close = np.isclose(reference, candidate, rtol=rtol, atol=atol)
    finite = np.isfinite(diff)
    return {
        "shape_match": True,
        "shape": list(reference.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "median_abs_error": float(np.median(diff)),
        "p99_abs_error": float(np.percentile(diff, 99)),
        "finite_fraction": float(finite.mean()),
        "close_fraction": float(close.mean()),
        "max_error_index": [int(value) for value in max_index],
        "reference_at_max_error": float(reference[max_index]),
        "candidate_at_max_error": float(candidate[max_index]),
        "rtol": rtol,
        "atol": atol,
        "allclose": bool(np.allclose(reference, candidate, rtol=rtol, atol=atol)),
    }


def classify_drift(comparison: dict[str, Any], trt_mode: str) -> str:
    if not comparison.get("shape_match", False):
        return "shape mismatch; check tensor names, dynamic profile shape, and engine/model pairing"
    if comparison.get("allclose", False):
        return "within tolerance; keep the evidence with the benchmark report"
    max_error = float(comparison.get("max_abs_error", 0.0))
    close_fraction = float(comparison.get("close_fraction", 0.0))
    if trt_mode == "engine":
        return (
            "outside tolerance for the serialized engine; check whether this engine precision, tactic set, "
            "or build input matches the ONNX model"
        )
    if close_fraction > 0.999 and max_error < 1e-2:
        return "small tail drift; review tolerance against detection quality before treating it as a bug"
    return "outside tolerance; debug preprocessing, export, unsupported ops, or TensorRT precision behavior"


def write_precision_note(report: dict[str, Any], path: Path) -> None:
    comparison = report.get("comparison")
    lines = [
        "# 06a Precision Alignment Note",
        "",
        f"- ONNX model: `{report['onnx']}`",
        f"- TensorRT mode: `{report['trt_mode']}`",
        f"- Input data: `{report['inputs']}`",
        f"- Scope: `{report['scope']}`",
        f"- ONNX Runtime output artifact: `{report['onnxrt_outputs']}`",
        f"- TensorRT comparison artifact: `{report.get('trt_outputs', 'skipped')}`",
        "",
        "## Scope",
        "",
        f"- {report['scope_note']}",
        "- Use lesson 12 to compare FP32, FP16, and INT8 detections across a representative image set.",
        "",
        "## Result",
        "",
    ]

    if comparison is None:
        lines.append("- TensorRT comparison was skipped. ONNX Runtime smoke output was generated successfully.")
    else:
        lines.extend(
            [
                f"- Output: `{comparison['output_name']}`",
                f"- Shape match: `{comparison['metrics']['shape_match']}`",
                f"- Allclose: `{comparison['metrics']['allclose']}` with rtol `{comparison['metrics']['rtol']}` and atol `{comparison['metrics']['atol']}`",
                f"- Max absolute error: `{comparison['metrics'].get('max_abs_error', 'n/a')}`",
                f"- Mean absolute error: `{comparison['metrics'].get('mean_abs_error', 'n/a')}`",
                f"- P99 absolute error: `{comparison['metrics'].get('p99_abs_error', 'n/a')}`",
                f"- Likely cause: {comparison['likely_cause']}",
            ]
        )

    lines.extend(
        [
            "",
            "## Debug Checklist",
            "",
            "- Confirm the same preprocessed tensor was used for both runners.",
            "- Confirm the ONNX model and TensorRT engine were built from the same export.",
            "- If FP16 is used, loosen tolerance only after checking detection quality on later validation images.",
            "- If drift starts at a specific tensor, use Polygraphy tensorwise inspection before changing postprocessing.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.onnx = args.onnx.resolve()
    args.engine = args.engine.resolve()
    args.inputs = args.inputs.resolve()
    args.output_dir = args.output_dir.resolve()

    try:
        validate_args(args)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        commands: list[CommandRecord] = []

        onnx_inspect_log = args.output_dir / "inspect_onnx.log"
        commands.append(
            run_command(
                "inspect_onnx",
                inspect_command(args.onnx, "onnx", onnx_inspect_log),
                onnx_inspect_log,
                keep_going=args.keep_going,
            )
        )

        onnxrt_outputs = args.output_dir / "onnxrt_outputs.json"
        onnxrt_log = args.output_dir / "run_onnxrt.log"
        commands.append(
            run_command(
                "run_onnxrt",
                onnxrt_command(args, onnxrt_outputs, onnxrt_log),
                onnxrt_log,
                keep_going=args.keep_going,
            )
        )

        comparison: dict[str, Any] | None = None
        trt_outputs: Path | None = None
        if not args.skip_trt:
            if args.trt_mode == "engine":
                engine_inspect_log = args.output_dir / "inspect_engine.log"
                commands.append(
                    run_command(
                        "inspect_engine",
                        inspect_command(args.engine, "engine", engine_inspect_log),
                        engine_inspect_log,
                        keep_going=args.keep_going,
                    )
                )

            trt_outputs = args.output_dir / "trt_compare_outputs.json"
            compare_log = args.output_dir / "compare_onnxrt_trt.log"
            commands.append(
                run_command(
                    "compare_onnxrt_trt",
                    compare_command(args, trt_outputs, compare_log),
                    compare_log,
                    keep_going=True,
                )
            )

            ort_runner, ort_output_name, ort_output = first_output(load_runner_outputs(onnxrt_outputs), "onnx")
            trt_runner, trt_output_name, trt_output = first_output(load_runner_outputs(trt_outputs), "trt")
            metrics = compare_arrays(ort_output, trt_output, args.rtol, args.atol)
            comparison = {
                "reference_runner": ort_runner,
                "candidate_runner": trt_runner,
                "output_name": ort_output_name,
                "candidate_output_name": trt_output_name,
                "metrics": metrics,
                "likely_cause": classify_drift(metrics, args.trt_mode),
            }

        report = {
            "onnx": str(args.onnx),
            "engine": str(args.engine) if args.engine.exists() else None,
            "inputs": str(args.inputs),
            "scope": REPORT_SCOPE,
            "scope_note": REPORT_SCOPE_NOTE,
            "trt_mode": args.trt_mode,
            "skip_trt": args.skip_trt,
            "rtol": args.rtol,
            "atol": args.atol,
            "onnxrt_outputs": str(onnxrt_outputs),
            "trt_outputs": str(trt_outputs) if trt_outputs is not None else None,
            "commands": [
                {
                    "name": record.name,
                    "command": record.command,
                    "log": str(record.log_path),
                    "return_code": record.return_code,
                }
                for record in commands
            ],
            "comparison": comparison,
        }

        report_path = args.output_dir / "precision_report.json"
        note_path = args.output_dir / "precision_alignment_note.md"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        write_precision_note(report, note_path)

        print(f"\nreport: {report_path}")
        print(f"note: {note_path}")
        if comparison is not None:
            metrics = comparison["metrics"]
            print(f"allclose(rtol={args.rtol}, atol={args.atol}): {metrics['allclose']}")
            print(f"max abs error: {metrics.get('max_abs_error', 'n/a')}")
            print(f"likely cause: {comparison['likely_cause']}")
            compare_record = next(record for record in commands if record.name == "compare_onnxrt_trt")
            if compare_record.return_code != 0 and not args.keep_going:
                return compare_record.return_code
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
