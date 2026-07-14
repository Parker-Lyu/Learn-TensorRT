#include "image_pipeline.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    lesson13::PipelineConfig pipeline;
    std::vector<std::filesystem::path> images;
};

std::size_t parse_size(const std::string& text, const std::string& option, bool allow_zero = false) {
    std::size_t consumed = 0;
    unsigned long long value = 0;
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(option + " requires a non-negative integer");
    }
    try {
        value = std::stoull(text, &consumed);
    } catch (const std::exception&) {
        throw std::invalid_argument(option + " requires a non-negative integer");
    }
    if (consumed != text.size() || (!allow_zero && value == 0) ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(option + " has an invalid value: " + text);
    }
    return static_cast<std::size_t>(value);
}

lesson13::OverloadPolicy parse_policy(const std::string& value) {
    if (value == "block") {
        return lesson13::OverloadPolicy::Block;
    }
    if (value == "drop-newest") {
        return lesson13::OverloadPolicy::DropNewest;
    }
    if (value == "drop-oldest") {
        return lesson13::OverloadPolicy::DropOldest;
    }
    throw std::invalid_argument("--policy must be block, drop-newest, or drop-oldest");
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument(arg + " requires a value");
            }
            return argv[++i];
        };

        if (arg == "--image") {
            options.images.emplace_back(require_value());
        } else if (arg == "--frames") {
            options.pipeline.frame_count = parse_size(require_value(), arg);
        } else if (arg == "--queue-capacity") {
            options.pipeline.queue_capacity = parse_size(require_value(), arg);
        } else if (arg == "--producer-delay-ms") {
            options.pipeline.producer_delay = std::chrono::milliseconds(
                parse_size(require_value(), arg, true));
        } else if (arg == "--consumer-delay-ms") {
            options.pipeline.consumer_delay = std::chrono::milliseconds(
                parse_size(require_value(), arg, true));
        } else if (arg == "--policy") {
            options.pipeline.overload_policy = parse_policy(require_value());
        } else if (arg == "--fail-producer-at") {
            options.pipeline.fail_producer_at = parse_size(require_value(), arg);
        } else if (arg == "--fail-consumer-at") {
            options.pipeline.fail_consumer_at = parse_size(require_value(), arg);
        } else if (arg == "--help") {
            std::cout
                << "Usage: cpp_producer_consumer [options]\n"
                << "  --image PATH                 Add an image source (repeatable)\n"
                << "  --frames N                   Frames to produce (default: 20)\n"
                << "  --queue-capacity N           Maximum queued frames (default: 4)\n"
                << "  --policy POLICY              block|drop-newest|drop-oldest\n"
                << "  --producer-delay-ms N        Delay after each read (default: 10)\n"
                << "  --consumer-delay-ms N        Simulated inference time (default: 40)\n"
                << "  --fail-producer-at N         Inject a producer error for study\n"
                << "  --fail-consumer-at N         Inject a consumer error for study\n";
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::invalid_argument("unknown option: " + arg);
        }
    }
    if (options.images.empty()) {
        const auto executable_dir = std::filesystem::absolute(argv[0]).parent_path();
        const auto assets_dir = (executable_dir / ".." / ".." / "assets").lexically_normal();
        options.images = {assets_dir / "dog.webp", assets_dir / "img2.jpeg"};
    }
    return options;
}

const char* policy_name(lesson13::OverloadPolicy policy) {
    switch (policy) {
        case lesson13::OverloadPolicy::Block:
            return "block";
        case lesson13::OverloadPolicy::DropNewest:
            return "drop-newest";
        case lesson13::OverloadPolicy::DropOldest:
            return "drop-oldest";
    }
    return "unknown";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        lesson13::ImagePipeline pipeline(options.pipeline);
        const lesson13::PipelineStats stats = pipeline.run(options.images);

        std::cout << std::fixed << std::setprecision(2)
                  << "policy: " << policy_name(options.pipeline.overload_policy) << '\n'
                  << "frames read: " << stats.frames_read << '\n'
                  << "frames processed: " << stats.frames_processed << '\n'
                  << "frames dropped: " << stats.queue.dropped << '\n'
                  << "queue high watermark: " << stats.queue.high_watermark << "/"
                  << options.pipeline.queue_capacity << '\n'
                  << "average queue latency: " << stats.average_queue_latency_ms << " ms\n"
                  << "max queue latency: " << stats.max_queue_latency_ms << " ms\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
