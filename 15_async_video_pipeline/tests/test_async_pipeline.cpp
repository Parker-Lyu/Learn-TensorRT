#include "async_pipeline.hpp"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }

int main() {
    try {
        lesson15::PipelineConfig config;
        config.queue_capacity = 3;
        config.max_batch_size = 4;
        config.simulated_inference = std::chrono::milliseconds(2);
        lesson15::AsyncVideoPipeline normal(config,
            std::make_unique<lesson15::SyntheticSource>(100, cv::Size(32, 24)));
        const auto metrics = normal.run();
        require(metrics.queue_high_watermark <= config.queue_capacity, "queue exceeded capacity");
        require(metrics.processed + metrics.dropped == metrics.captured, "counter accounting mismatch");
        require(metrics.latency_p99_ms >= metrics.latency_p50_ms, "percentiles are not ordered");

        config.fail_worker_at = 2;
        bool failed = false;
        try {
            lesson15::AsyncVideoPipeline broken(config,
                std::make_unique<lesson15::SyntheticSource>(20, cv::Size(32, 24)));
            (void)broken.run();
        } catch (const std::exception&) { failed = true; }
        require(failed, "worker failure was not propagated");

        config.fail_worker_at = 0;
        config.simulated_inference = std::chrono::milliseconds(10);
        lesson15::AsyncVideoPipeline stopped(config,
            std::make_unique<lesson15::SyntheticSource>(10000, cv::Size(32, 24)));
        auto run = std::async(std::launch::async, [&] { return stopped.run(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        stopped.stop();
        require(run.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
                "explicit stop deadlocked");
        (void)run.get();
        std::cout << "All async pipeline tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
