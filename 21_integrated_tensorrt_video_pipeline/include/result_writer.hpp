#pragma once

#include "tensorrt_backend.hpp"

#include <opencv2/core.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

namespace lesson21 {

struct ResultTiming {
    double cpu_postprocess_ms{};
    double result_output_ms{};
};

class ResultWriter {
public:
    ResultWriter(const std::filesystem::path& output_directory,
                 std::size_t maximum_detection_records);
    ResultTiming write(const GpuBatchResult& result, const std::vector<cv::Mat>& source_images,
                 std::vector<double>& frame_latencies_ms);

private:
    std::filesystem::path output_directory_;
    std::ofstream detections_;
    bool annotation_written_{false};
    std::size_t maximum_detection_records_;
    std::size_t detection_records_written_{0};
};

}  // namespace lesson21
