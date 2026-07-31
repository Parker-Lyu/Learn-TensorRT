#pragma once
#include "pipeline_core.hpp"
#include <opencv2/core.hpp>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
namespace lesson21 {
struct RuntimeIdentity { std::string gpu_name; int compute_major{},compute_minor{},tensorrt_major{},tensorrt_minor{},tensorrt_patch{},cuda_runtime{},cuda_driver{}; };
struct GpuBatchResult { BatchMetadata metadata; std::vector<float> output; std::vector<int64_t> output_shape; float preprocess_ms{}, inference_ms{}, d2h_ms{}; };
class TensorRtBackend {
public:
  TensorRtBackend(const std::string& engine, std::size_t slots, cv::Size input_size);
  ~TensorRtBackend();
  TensorRtBackend(const TensorRtBackend&) = delete;
  void submit(std::size_t slot, const std::vector<cv::Mat>& images, BatchMetadata metadata);
  bool ready(std::size_t slot) const;
  GpuBatchResult collect(std::size_t slot);
  RuntimeIdentity identity() const;
private: struct Impl; std::unique_ptr<Impl> impl_;
};
}
