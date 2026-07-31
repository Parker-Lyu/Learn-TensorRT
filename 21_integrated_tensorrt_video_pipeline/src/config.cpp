#include "config.hpp"

#include "frame_scheduler.hpp"

#include <sstream>
#include <stdexcept>

namespace lesson21 {
namespace {

std::size_t positive(const char* value, const char* name) {
    const long long parsed = std::stoll(value);
    if (parsed <= 0) throw std::invalid_argument(std::string(name) + " must be positive");
    return static_cast<std::size_t>(parsed);
}

}  // namespace

std::string usage() {
    return "usage: integrated_tensorrt_video_pipeline_gpu ENGINE SOURCE[,SOURCE...] "
           "[FRAMES=16] [BATCH=4] [SLOTS=2] [OUTPUT] [block|drop-oldest] "
           "[QUEUE_CAPACITY=4] [CAPTURE_INTERVAL_MS=0] [round-robin|latest-first] "
           "[--duration-seconds N] [--repeat-source] [--metrics-interval-seconds N] "
           "[--max-detection-records N]";
}

PipelineConfig parse_config(int argc, char** argv) {
    if (argc < 3) throw std::invalid_argument(usage());
    PipelineConfig config;
    config.scheduling_policy = SchedulingPolicy::RoundRobin;
    config.engine_path = argv[1];
    std::stringstream source_list(argv[2]);
    std::string source;
    while (std::getline(source_list, source, ',')) {
        if (!source.empty()) config.source_paths.push_back(source);
    }
    if (config.source_paths.empty()) throw std::invalid_argument("no sources were provided");

    int index = 3;
    if (index < argc && argv[index][0] != '-') config.frame_count = positive(argv[index++], "frames");
    if (index < argc && argv[index][0] != '-') config.maximum_batch = positive(argv[index++], "batch");
    if (index < argc && argv[index][0] != '-') config.slot_count = positive(argv[index++], "slots");
    if (index < argc && argv[index][0] != '-') config.output_directory = argv[index++];
    if (index < argc && argv[index][0] != '-') {
        config.overload_name = argv[index++];
        if (config.overload_name == "block") config.overload_policy = OverloadPolicy::Block;
        else if (config.overload_name == "drop-oldest") {
            config.overload_policy = OverloadPolicy::DropOldest;
        } else throw std::invalid_argument("overload policy must be block or drop-oldest");
    }
    if (index < argc && argv[index][0] != '-') {
        config.queue_capacity = positive(argv[index++], "queue capacity");
    }
    if (index < argc && argv[index][0] != '-') {
        const long long milliseconds = std::stoll(argv[index++]);
        if (milliseconds < 0) throw std::invalid_argument("capture interval cannot be negative");
        config.capture_interval = std::chrono::milliseconds(milliseconds);
    }
    if (index < argc && argv[index][0] != '-') {
        config.scheduling_name = argv[index++];
        if (config.scheduling_name == "round-robin") {
            config.scheduling_policy = SchedulingPolicy::RoundRobin;
        } else if (config.scheduling_name == "latest-first") {
            config.scheduling_policy = SchedulingPolicy::LatestFirst;
        } else throw std::invalid_argument("scheduling policy must be round-robin or latest-first");
    }
    while (index < argc) {
        const std::string option = argv[index++];
        if (option == "--repeat-source") {
            config.repeat_source = true;
        } else if (option == "--duration-seconds" && index < argc) {
            config.duration = std::chrono::seconds(positive(argv[index++], "duration"));
        } else if (option == "--metrics-interval-seconds" && index < argc) {
            config.metrics_interval = std::chrono::seconds(
                positive(argv[index++], "metrics interval"));
        } else if (option == "--max-detection-records" && index < argc) {
            config.maximum_detection_records = positive(argv[index++], "maximum detection records");
        } else {
            throw std::invalid_argument("unknown or incomplete option: " + option);
        }
    }
    if (config.maximum_batch > 4) throw std::invalid_argument("batch must be in range 1..4");
    if (config.duration.count() != 0 && !config.repeat_source) {
        throw std::invalid_argument("duration mode requires --repeat-source");
    }
    return config;
}

}  // namespace lesson21
