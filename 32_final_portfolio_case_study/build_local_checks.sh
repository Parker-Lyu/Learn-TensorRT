#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

lessons=(
  11_yolov8_trt_cpp
  16_cpp_producer_consumer
  17_dynamic_batching
  18_async_video_pipeline
  19_multistream_video_pipeline
  20_cuda_preprocess_npp
  29_cpp_shared_library_python_binding
  31_cpp_interview_katas
)

for lesson in "${lessons[@]}"; do
  rm -rf "${ROOT_DIR}/${lesson}/build"
  cmake -S "${ROOT_DIR}/${lesson}" -B "${ROOT_DIR}/${lesson}/build" \
    -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build "${ROOT_DIR}/${lesson}/build" -j"$(nproc)"
done
