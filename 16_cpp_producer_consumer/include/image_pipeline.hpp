#pragma once

#include "bounded_queue.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <mutex>
#include <vector>

namespace lesson16 {

struct ImageFrame {
    std::size_t sequence{0};
    std::filesystem::path source_path;
    std::vector<std::uint8_t> encoded_bytes;
    std::chrono::steady_clock::time_point captured_at;
};

struct PipelineConfig {
    std::size_t queue_capacity{4};
    std::size_t frame_count{20};
    OverloadPolicy overload_policy{OverloadPolicy::DropOldest};
    std::chrono::milliseconds producer_delay{10};
    std::chrono::milliseconds consumer_delay{40};
    // Test hooks also demonstrate that worker failures are not silently lost.
    std::size_t fail_producer_at{0};
    std::size_t fail_consumer_at{0};
};

struct PipelineStats {
    std::size_t frames_read{0};
    std::size_t frames_processed{0};
    QueueStats queue;
    double average_queue_latency_ms{0.0};
    double max_queue_latency_ms{0.0};
};

class ImagePipeline {
public:
    explicit ImagePipeline(PipelineConfig config);
    ~ImagePipeline();

    ImagePipeline(const ImagePipeline&) = delete;
    ImagePipeline& operator=(const ImagePipeline&) = delete;

    PipelineStats run(const std::vector<std::filesystem::path>& image_paths);
    void stop() noexcept;

private:
    void producer_loop(const std::vector<std::filesystem::path>& image_paths);
    void consumer_loop();
    void record_failure(std::exception_ptr failure) noexcept;

    PipelineConfig config_;
    BoundedQueue<ImageFrame> queue_;
    std::atomic<bool> stop_requested_{false};
    std::atomic<std::size_t> frames_read_{0};
    std::atomic<std::size_t> frames_processed_{0};
    std::mutex failure_mutex_;
    std::exception_ptr failure_;
    std::vector<double> queue_latency_ms_;
    bool run_called_{false};
};

}  // namespace lesson16
