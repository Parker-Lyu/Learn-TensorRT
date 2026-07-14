#include "multistream_pipeline.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        lesson16::PipelineConfig config;
        std::vector<std::string> inputs;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            auto value = [&]() { if (++index >= argc) throw std::invalid_argument(argument + " requires a value"); return std::string(argv[index]); };
            if (argument == "--input") inputs.push_back(value());
            else if (argument == "--queue-capacity") config.queue_capacity = std::stoull(value());
            else if (argument == "--max-batch") config.max_batch_size = std::stoull(value());
            else if (argument == "--scheduler") {
                const auto policy = value();
                if (policy == "round-robin") config.scheduling = lesson16::SchedulingPolicy::RoundRobin;
                else if (policy == "latest") config.scheduling = lesson16::SchedulingPolicy::LatestFirst;
                else throw std::invalid_argument("scheduler must be round-robin or latest");
            } else if (argument == "--fail-inference-batch") config.fail_inference_batch = std::stoull(value());
            else if (argument == "--help") {
                std::cout << "Usage: multistream_video_pipeline [--input VIDEO ...] [--scheduler round-robin|latest] "
                             "[--queue-capacity N] [--max-batch N] [--fail-inference-batch N]\n";
                return EXIT_SUCCESS;
            } else throw std::invalid_argument("unknown option: " + argument);
        }
        std::vector<lesson16::StreamConfig> streams;
        if (inputs.empty()) streams = {{0, "", 120, std::chrono::milliseconds(2)},
                                       {1, "", 80, std::chrono::milliseconds(4)}};
        else {
            if (inputs.size() < 2) throw std::invalid_argument("provide at least two --input values");
            for (std::size_t index = 0; index < inputs.size(); ++index)
                streams.push_back({index, inputs[index], 0, std::chrono::milliseconds(0)});
        }
        lesson16::MultiStreamPipeline pipeline(config, std::move(streams));
        const auto report = pipeline.run();
        std::cout << std::fixed << std::setprecision(2) << "total_fps=" << report.total_fps << '\n';
        for (const auto& stream : report.streams) {
            std::cout << "stream=" << stream.stream_id << " captured=" << stream.captured
                      << " processed=" << stream.processed << " dropped=" << stream.dropped
                      << " queue_peak=" << stream.queue_high_watermark << " fps=" << stream.fps
                      << " p50=" << stream.latency_p50_ms << " p90=" << stream.latency_p90_ms
                      << " p99=" << stream.latency_p99_ms << '\n';
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
