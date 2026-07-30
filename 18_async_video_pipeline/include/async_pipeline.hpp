#pragma once

#include <opencv2/core.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace lesson18 {

struct Frame {
    std::size_t id{0};
    cv::Mat image;
    std::chrono::steady_clock::time_point captured_at;
};

class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual std::optional<cv::Mat> read() = 0;
};

class VideoSource final : public FrameSource {
public:
    explicit VideoSource(const std::string& uri);
    ~VideoSource();
    std::optional<cv::Mat> read() override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class SyntheticSource final : public FrameSource {
public:
    SyntheticSource(std::size_t frame_count, cv::Size size = {640, 480});
    std::optional<cv::Mat> read() override;
private:
    std::size_t remaining_;
    std::size_t sequence_{0};
    cv::Size size_;
};

struct PipelineConfig {
    std::size_t queue_capacity{8};
    std::size_t max_batch_size{4};
    std::chrono::milliseconds batch_timeout{5};
    std::chrono::milliseconds simulated_inference{8};
    std::size_t fail_capture_at{0};
    std::size_t fail_worker_at{0};
};

struct PipelineMetrics {
    std::size_t captured{0};
    std::size_t processed{0};
    std::size_t dropped{0};
    std::size_t queue_high_watermark{0};
    double elapsed_seconds{0.0};
    double fps{0.0};
    double latency_p50_ms{0.0};
    double latency_p90_ms{0.0};
    double latency_p99_ms{0.0};
};

class AsyncVideoPipeline {
public:
    AsyncVideoPipeline(PipelineConfig config, std::unique_ptr<FrameSource> source);
    ~AsyncVideoPipeline();
    PipelineMetrics run();
    void stop() noexcept;
private:
    class FrameQueue;
    void capture_loop();
    void inference_loop();
    void fail(std::exception_ptr error) noexcept;

    PipelineConfig config_;
    std::unique_ptr<FrameSource> source_;
    std::unique_ptr<FrameQueue> queue_;
    std::atomic<bool> cancelled_{false};
    std::atomic<std::size_t> captured_{0};
    std::atomic<std::size_t> processed_{0};
    std::mutex result_mutex_;
    std::vector<double> latencies_ms_;
    std::mutex failure_mutex_;
    std::exception_ptr failure_;
    bool ran_{false};
};

}  // namespace lesson18
