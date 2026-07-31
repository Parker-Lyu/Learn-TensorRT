#pragma once

#include "pipeline_core.hpp"

#include <chrono>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

namespace lesson21 {

enum class SchedulingPolicy;

struct PipelineConfig {
    std::string engine_path;
    std::vector<std::string> source_paths;
    std::size_t frame_count{16};
    std::size_t maximum_batch{4};
    std::size_t slot_count{2};
    std::filesystem::path output_directory{"21_integrated_tensorrt_video_pipeline/output"};
    OverloadPolicy overload_policy{OverloadPolicy::Block};
    std::string overload_name{"block"};
    std::size_t queue_capacity{4};
    std::chrono::milliseconds capture_interval{0};
    SchedulingPolicy scheduling_policy;
    std::string scheduling_name{"round-robin"};
    std::chrono::seconds duration{0};
    bool repeat_source{false};
    std::chrono::seconds metrics_interval{5};
    std::size_t maximum_detection_records{std::numeric_limits<std::size_t>::max()};
};

PipelineConfig parse_config(int argc, char** argv);
std::string usage();

}  // namespace lesson21
