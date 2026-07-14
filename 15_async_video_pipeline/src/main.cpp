#include "async_pipeline.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        lesson15::PipelineConfig config;
        std::string input;
        std::size_t synthetic_frames = 120;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            auto value = [&]() { if (++index >= argc) throw std::invalid_argument(argument + " requires a value"); return std::string(argv[index]); };
            if (argument == "--input") input = value();
            else if (argument == "--synthetic-frames") synthetic_frames = std::stoull(value());
            else if (argument == "--queue-capacity") config.queue_capacity = std::stoull(value());
            else if (argument == "--max-batch") config.max_batch_size = std::stoull(value());
            else if (argument == "--batch-timeout-ms") config.batch_timeout = std::chrono::milliseconds(std::stoll(value()));
            else if (argument == "--inference-ms") config.simulated_inference = std::chrono::milliseconds(std::stoll(value()));
            else if (argument == "--fail-capture-at") config.fail_capture_at = std::stoull(value());
            else if (argument == "--fail-worker-at") config.fail_worker_at = std::stoull(value());
            else if (argument == "--help") {
                std::cout << "Usage: async_video_pipeline [--input VIDEO_OR_CAMERA] [--synthetic-frames N] "
                             "[--queue-capacity N] [--max-batch N] [--batch-timeout-ms N] "
                             "[--inference-ms N] [--fail-capture-at N] [--fail-worker-at N]\n";
                return EXIT_SUCCESS;
            } else throw std::invalid_argument("unknown option: " + argument);
        }
        std::unique_ptr<lesson15::FrameSource> source;
        if (input.empty()) source = std::make_unique<lesson15::SyntheticSource>(synthetic_frames);
        else source = std::make_unique<lesson15::VideoSource>(input);
        lesson15::AsyncVideoPipeline pipeline(config, std::move(source));
        const auto metrics = pipeline.run();
        std::cout << std::fixed << std::setprecision(2)
                  << "captured=" << metrics.captured << " processed=" << metrics.processed
                  << " dropped=" << metrics.dropped << " queue_peak=" << metrics.queue_high_watermark
                  << " fps=" << metrics.fps << " p50_ms=" << metrics.latency_p50_ms
                  << " p90_ms=" << metrics.latency_p90_ms << " p99_ms=" << metrics.latency_p99_ms
                  << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
