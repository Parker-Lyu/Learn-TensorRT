#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKS = [
    ("lesson10 preprocessing/postprocessing",
     ["10_yolov8_trt_cpp/build/yolov8_cpp_tests"]),
    ("lesson13 concurrency",
     ["ctest", "--test-dir", "13_cpp_producer_consumer/build", "--output-on-failure"]),
    ("lesson14 batching",
     ["ctest", "--test-dir", "14_dynamic_batching/build", "--output-on-failure"]),
    ("lesson15 async pipeline",
     ["ctest", "--test-dir", "15_async_video_pipeline/build", "--output-on-failure"]),
    ("lesson16 multistream",
     ["ctest", "--test-dir", "16_multistream_video_pipeline/build", "--output-on-failure"]),
    ("lesson17 CUDA preprocess",
     ["ctest", "--test-dir", "17_cuda_preprocess_npp/build", "--output-on-failure"]),
    ("lesson21 ctypes inference",
     [sys.executable, "-m", "unittest", "discover", "-s",
      "21_cpp_shared_library_python_binding/tests", "-v"]),
    ("lesson23 katas",
     ["ctest", "--test-dir", "23_cpp_interview_katas/build", "--output-on-failure"]),
]


def run(command: list[str]) -> dict:
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    except OSError as error:
        return {"command": command, "returncode": None,
                "elapsed_seconds": time.monotonic() - started,
                "stdout": "", "stderr": str(error)}
    return {"command": command, "returncode": result.returncode,
            "elapsed_seconds": time.monotonic() - started,
            "stdout": result.stdout, "stderr": result.stderr}


def platform_identity() -> dict:
    gpu = run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
               "--format=csv,noheader,nounits"])
    tensorrt = run([sys.executable, "-c", "import tensorrt; print(tensorrt.__version__)"])
    cuda = run(["nvcc", "--version"])
    return {
        "development_image": os.environ.get(
            "LEARN_TENSORRT_DEVELOPMENT_IMAGE", "nvcr.io/nvidia/pytorch:25.11-py3"),
        "container_build_id": os.environ.get("NVIDIA_BUILD_ID", "not exposed"),
        "declared_gpu": os.environ.get("LEARN_TENSORRT_GPU", "NVIDIA GeForce RTX 4090"),
        "gpu": gpu,
        "tensorrt": tensorrt,
        "cuda_toolkit": cuda,
        "host": {"system": platform.system(), "release": platform.release(),
                 "machine": platform.machine(), "cpu_count": os.cpu_count()},
    }


def main() -> int:
    results = []
    for name, command in CHECKS:
        result = run(command)
        result["name"] = name
        results.append(result)

    evidence = {
        "schema_version": 2,
        "platform": platform_identity(),
        "checks": results,
        "passed": all(item["returncode"] == 0 for item in results),
    }
    output = ROOT / "24_final_portfolio_case_study/outputs/local_checks.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": evidence["passed"],
                      "checks": [{"name": item["name"], "returncode": item["returncode"]}
                                 for item in results]}, indent=2))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
