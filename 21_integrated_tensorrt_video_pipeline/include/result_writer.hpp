#pragma once

#include "tensorrt_backend.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

namespace lesson21 {

class ResultWriter {
public:
    explicit ResultWriter(const std::filesystem::path& output_directory);
    double write(const GpuBatchResult& result, const std::vector<cv::Mat>& source_images,
                 std::vector<double>& frame_latencies_ms);

private:
    std::filesystem::path output_directory_;
    std::ofstream detections_;
    bool annotation_written_{false};
};

}  // namespace lesson21
