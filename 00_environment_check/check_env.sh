#!/usr/bin/env bash
set -u

failures=0

section() {
  printf '\n========== %s ==========\n' "$1"
}

check_cmd() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    printf '[OK] %s: %s\n' "$name" "$(command -v "$name")"
  else
    printf '[MISSING] %s\n' "$name"
    failures=$((failures + 1))
  fi
}

run_check() {
  local label="$1"
  shift
  printf '\n-- %s --\n' "$label"
  if "$@"; then
    printf '[OK] %s\n' "$label"
  else
    local code=$?
    printf '[FAIL] %s (exit=%s)\n' "$label" "$code"
    failures=$((failures + 1))
  fi
}

run_optional() {
  local label="$1"
  shift
  printf '\n-- %s --\n' "$label"
  if "$@"; then
    printf '[OK] %s\n' "$label"
  else
    local code=$?
    printf '[WARN] %s (exit=%s)\n' "$label" "$code"
  fi
}

section "System"
run_optional "OS release" sh -c 'cat /etc/os-release'
run_optional "Kernel" uname -a
run_optional "Working directory" pwd

section "Required commands"
for cmd in nvidia-smi nvcc trtexec cmake g++ python3; do
  check_cmd "$cmd"
done

section "GPU"
run_check "nvidia-smi" nvidia-smi

section "CUDA"
run_check "nvcc --version" nvcc --version

section "TensorRT"
run_optional "trtexec help/version output" sh -c 'trtexec --help 2>&1 | head -n 20'
run_check "TensorRT C++ libraries" sh -c 'ldconfig -p 2>/dev/null | grep -q libnvinfer || ls /usr/lib/x86_64-linux-gnu/libnvinfer.so* >/dev/null 2>&1'
run_optional "TensorRT Python import" python3 - <<'PY'
try:
    import tensorrt as trt
    print("tensorrt:", trt.__version__)
except Exception as exc:
    raise SystemExit(exc)
PY

section "C++ build tools"
run_check "cmake --version" cmake --version
run_check "g++ --version" g++ --version

section "Python"
run_check "python3 --version" python3 --version
run_check "Required Python package imports" python3 - <<'PY'
import importlib.util

required = [
    "onnx",
    "onnxruntime",
]

missing = []
for package in required:
    spec = importlib.util.find_spec(package)
    print(f"{package}: {'installed' if spec else 'missing'}")
    if spec is None:
        missing.append(package)

if missing:
    raise SystemExit("missing required package(s): " + ", ".join(missing))
PY

run_optional "Optional Python package availability" python3 - <<'PY'
import importlib.util

packages = [
    "cv2",
    "numpy",
    "torch",
    "tensorrt",
]

for package in packages:
    spec = importlib.util.find_spec(package)
    print(f"{package}: {'installed' if spec else 'missing'}")
PY

section "OpenCV"
run_optional "pkg-config opencv4" sh -c 'pkg-config --modversion opencv4'
run_optional "Python OpenCV import" python3 - <<'PY'
try:
    import cv2
    print("cv2:", cv2.__version__)
except Exception as exc:
    raise SystemExit(exc)
PY

section "Project mount"
run_check "Repository is writable" sh -c 'test -w .'
run_optional "Git status" git status --short

section "Summary"
if [ "$failures" -eq 0 ]; then
  printf '[PASS] Environment checks passed.\n'
  exit 0
fi

printf '[FAIL] Environment checks finished with %s required failure(s).\n' "$failures"
exit 1
