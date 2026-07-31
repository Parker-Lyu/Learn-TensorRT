#pragma once

#include "pipeline_core.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lesson21 {

struct RuntimeIdentity {
    std::string gpu_name;
    int compute_major{};
    int compute_minor{};
    int tensorrt_major{};
    int tensorrt_minor{};
    int tensorrt_patch{};
    int cuda_runtime{};
    int cuda_driver{};
};

struct GpuBatchResult {
    BatchMetadata metadata;
    std::vector<float> output;
    std::vector<int64_t> output_shape;
    double host_staging_ms{};
    double capacity_growth_ms{};
    float h2d_ms{};
    float preprocess_ms{};
    float inference_ms{};
    float d2h_ms{};
};

class TensorRtBackend {
public:
    TensorRtBackend(const std::string& engine, std::size_t slots, cv::Size input_size);
    ~TensorRtBackend();
    TensorRtBackend(const TensorRtBackend&) = delete;
    TensorRtBackend& operator=(const TensorRtBackend&) = delete;

    std::optional<std::size_t> try_reserve();
    std::size_t reserve();
    void submit(std::size_t slot, const std::vector<cv::Mat>& images, BatchMetadata metadata);
    bool ready(std::size_t slot) const;
    GpuBatchResult collect(std::size_t slot);
    RuntimeIdentity identity() const;
    std::size_t available_slots() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lesson21
