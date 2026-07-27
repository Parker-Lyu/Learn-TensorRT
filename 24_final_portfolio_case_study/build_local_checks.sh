#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

lessons=(
  10_yolov8_trt_cpp
  13_cpp_producer_consumer
  14_dynamic_batching
  15_async_video_pipeline
  16_multistream_video_pipeline
  17_cuda_preprocess_npp
  21_cpp_shared_library_python_binding
  23_cpp_interview_katas
)

for lesson in "${lessons[@]}"; do
  rm -rf "${ROOT_DIR}/${lesson}/build"
  cmake -S "${ROOT_DIR}/${lesson}" -B "${ROOT_DIR}/${lesson}/build" \
    -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build "${ROOT_DIR}/${lesson}/build" -j"$(nproc)"
done
