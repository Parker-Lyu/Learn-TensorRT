#!/usr/bin/env python3
"""Create and validate immutable reference-bundle identities for candidate reuse."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_IDENTITY_FIELDS = (
    "weights_sha256",
    "onnx_sha256",
    "validation_manifest_sha256",
    "quality_contract_sha256",
    "preprocessing_id",
    "postprocessing_id",
    "metric_id",
    "runtime_id",
    "fp32_engine_sha256",
    "fp16_engine_sha256",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_identity(identity: dict[str, Any]) -> dict[str, str]:
    missing = [field for field in REQUIRED_IDENTITY_FIELDS if not identity.get(field)]
    if missing:
        raise ValueError(f"reference identity is missing fields: {', '.join(missing)}")
    return {field: str(identity[field]) for field in REQUIRED_IDENTITY_FIELDS}


def reference_id(identity: dict[str, Any]) -> str:
    encoded = json.dumps(
        canonical_identity(identity), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def write_bundle(
    output: Path,
    identity: dict[str, Any],
    report_path: Path,
    environment: dict[str, Any],
) -> dict[str, Any]:
    if not report_path.is_file():
        raise FileNotFoundError(f"reference report not found: {report_path}")
    normalized = canonical_identity(identity)
    document = {
        "schema_version": 1,
        "reference_id": reference_id(normalized),
        "identity": normalized,
        "reference_report": str(report_path.resolve()),
        "reference_report_sha256": sha256(report_path),
        "environment": environment,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def load_bundle(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    identity = canonical_identity(document.get("identity", {}))
    expected_id = reference_id(identity)
    if document.get("reference_id") != expected_id:
        raise ValueError("reference bundle identity hash does not match its contents")
    report = Path(document["reference_report"])
    if not report.is_file():
        raise FileNotFoundError(f"reference report not found: {report}")
    if sha256(report) != document.get("reference_report_sha256"):
        raise ValueError("reference report hash does not match the bundle")
    return document


def assert_compatible(bundle: dict[str, Any], expected: dict[str, Any]) -> None:
    actual = canonical_identity(bundle.get("identity", {}))
    wanted = canonical_identity(expected)
    differences = [field for field in REQUIRED_IDENTITY_FIELDS if actual[field] != wanted[field]]
    if differences:
        raise ValueError("reference bundle is incompatible: " + ", ".join(differences))

