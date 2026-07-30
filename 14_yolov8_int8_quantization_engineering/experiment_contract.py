#!/usr/bin/env python3
"""Validate candidate engines against the declared Lesson 14 experiment matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


LESSON_DIR = Path(__file__).resolve().parent
REPO_ROOT = LESSON_DIR.parent
DEFAULT_EXPERIMENTS = LESSON_DIR / "configs/experiments.json"
DEFAULT_ENVIRONMENTS = LESSON_DIR / "configs/environments.json"
REPOSITORY_MARKER = "Learn-TensorRT"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_experiments(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError(f"unsupported experiment-matrix schema in {path}")
    stages = document.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("experiment matrix has no stages")
    ids = [stage.get("id") for stage in stages]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("experiment stage has no ID")
    if len(set(ids)) != len(ids):
        raise ValueError("experiment matrix contains duplicate stage IDs")
    return document


def experiment_stage(document: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    for stage in document["stages"]:
        if stage["id"] == experiment_id:
            return stage
    raise ValueError(f"unknown experiment ID: {experiment_id}")


def nested_value(document: dict[str, Any], dotted_name: str) -> Any:
    value: Any = document
    for part in dotted_name.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"engine metadata is missing required field: {dotted_name}")
        value = value[part]
    return value


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    if path.exists():
        return path
    if REPOSITORY_MARKER in path.parts:
        index = path.parts.index(REPOSITORY_MARKER)
        relocated = REPO_ROOT.joinpath(*path.parts[index + 1 :])
        if relocated.exists():
            return relocated
    return path


def engine_hash(metadata: dict[str, Any]) -> str:
    value = metadata.get("engine_sha256")
    if isinstance(value, str):
        return value
    engine = metadata.get("engine")
    if isinstance(engine, dict) and isinstance(engine.get("sha256"), str):
        return engine["sha256"]
    raise ValueError("engine metadata has no engine SHA-256")


def source_onnx(metadata: dict[str, Any]) -> tuple[Path, str] | None:
    if isinstance(metadata.get("onnx"), str):
        digest = metadata.get("onnx_sha256")
        if not isinstance(digest, str):
            raise ValueError("engine metadata has no ONNX SHA-256")
        return resolve_recorded_path(metadata["onnx"]), digest
    source = metadata.get("source_onnx")
    if isinstance(source, dict):
        if not isinstance(source.get("path"), str) or not isinstance(source.get("sha256"), str):
            raise ValueError("engine metadata has an incomplete source_onnx identity")
        return resolve_recorded_path(source["path"]), source["sha256"]
    return None


def validate_engine_for_experiment(
    experiment_id: str,
    experiments_path: Path,
    environments_path: Path,
    engine_path: Path,
    metadata_path: Path,
    manifest_path: Path,
    tensorrt_version: str,
) -> dict[str, Any]:
    experiments = load_experiments(experiments_path)
    stage = experiment_stage(experiments, experiment_id)
    environments = json.loads(environments_path.read_text(encoding="utf-8"))
    environment = environments.get("environments", {}).get(stage["runtime"])
    if not isinstance(environment, dict):
        raise ValueError(f"experiment runtime is not declared: {stage['runtime']}")
    if environment.get("tensorrt") != tensorrt_version:
        raise ValueError(
            f"experiment {experiment_id} requires TensorRT {environment.get('tensorrt')}, "
            f"found {tensorrt_version}"
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_engine_hash = sha256(engine_path)
    if engine_hash(metadata) != actual_engine_hash:
        raise ValueError("candidate engine hash does not match its build metadata")
    if metadata.get("tensorrt_version") != tensorrt_version:
        raise ValueError("candidate engine metadata records a different TensorRT version")

    expectations = stage.get("engine_metadata", {})
    for field, expected in expectations.items():
        actual = nested_value(metadata, field)
        if actual != expected:
            raise ValueError(
                f"experiment {experiment_id} metadata mismatch for {field}: "
                f"expected {expected!r}, found {actual!r}"
            )

    manifest_hash = sha256(manifest_path)
    if "manifest_sha256" in metadata and metadata["manifest_sha256"] != manifest_hash:
        raise ValueError("candidate engine metadata uses a different dataset manifest")

    source_identity = source_onnx(metadata)
    source_metadata_identity = None
    source_expectations = stage.get("source_model_metadata", {})
    if source_expectations:
        if source_identity is None:
            raise ValueError("experiment requires source-model evidence but engine metadata has none")
        onnx_path, declared_onnx_hash = source_identity
        if not onnx_path.is_file() or sha256(onnx_path) != declared_onnx_hash:
            raise ValueError("candidate source ONNX identity does not match engine metadata")
        source_metadata_path = onnx_path.with_suffix(onnx_path.suffix + ".json")
        if not source_metadata_path.is_file():
            raise FileNotFoundError(f"source-model metadata not found: {source_metadata_path}")
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata.get("onnx_sha256") != declared_onnx_hash:
            raise ValueError("source-model metadata records a different ONNX identity")
        if source_metadata.get("manifest_sha256") != manifest_hash:
            raise ValueError("source-model metadata uses a different dataset manifest")
        for field, expected in source_expectations.items():
            actual = nested_value(source_metadata, field)
            if actual != expected:
                raise ValueError(
                    f"experiment {experiment_id} source metadata mismatch for {field}: "
                    f"expected {expected!r}, found {actual!r}"
                )
        source_metadata_identity = {
            "path": str(source_metadata_path.resolve()),
            "sha256": sha256(source_metadata_path),
        }

    return {
        "id": experiment_id,
        "matrix": str(experiments_path.resolve()),
        "matrix_sha256": sha256(experiments_path),
        "stage": stage,
        "engine_metadata": {
            "path": str(metadata_path.resolve()),
            "sha256": sha256(metadata_path),
        },
        "source_model_metadata": source_metadata_identity,
    }
