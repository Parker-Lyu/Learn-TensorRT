#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

namespace lesson20 {

enum class HostMemoryMode { Pageable, Pinned, Mapped };

struct PreprocessTiming {
    float host_staging_ms{0.0F};
    float h2d_ms{0.0F};
    float gpu_preprocess_ms{0.0F};
    float d2h_ms{0.0F};
};

struct GpuPreprocessResult {
    std::vector<float> tensor_nchw;
    PreprocessTiming timing;
};

std::vector<float> cpu_resize_bgr_to_rgb_nchw(const cv::Mat& bgr, cv::Size target);

class GpuPreprocessor {
public:
    GpuPreprocessor(cv::Size source, cv::Size target, HostMemoryMode mode);
    ~GpuPreprocessor();
    GpuPreprocessor(const GpuPreprocessor&) = delete;
    GpuPreprocessor& operator=(const GpuPreprocessor&) = delete;
    GpuPreprocessResult run(const cv::Mat& bgr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lesson20
