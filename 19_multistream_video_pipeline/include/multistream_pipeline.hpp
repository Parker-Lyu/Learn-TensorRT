#pragma once

#include <opencv2/core.hpp>

#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lesson19 {

enum class SchedulingPolicy { RoundRobin, LatestFirst };
enum class SourceFailurePolicy { IsolateStream, StopAll };

struct StreamConfig {
    std::size_t stream_id{0};
    std::string uri;
    std::size_t synthetic_frames{100};
    std::chrono::milliseconds frame_interval{2};
    std::size_t fail_at_frame{0};
};

struct PipelineConfig {
    std::size_t queue_capacity{4};
    std::size_t max_batch_size{4};
    std::size_t max_in_flight_batches{2};
    std::chrono::milliseconds batch_timeout{4};
    std::chrono::milliseconds simulated_inference{3};
    SchedulingPolicy scheduling{SchedulingPolicy::RoundRobin};
    SourceFailurePolicy source_failure{SourceFailurePolicy::IsolateStream};
    std::size_t fail_inference_batch{0};
};

struct ResultIdentity {
    std::size_t stream_id{0};
    std::size_t frame_id{0};
};

struct StreamMetrics {
    std::size_t stream_id{0};
    std::size_t captured{0};
    std::size_t processed{0};
    std::size_t dropped{0};
    std::size_t source_failures{0};
    std::size_t queue_high_watermark{0};
    double fps{0.0};
    double latency_p50_ms{0.0};
    double latency_p90_ms{0.0};
    double latency_p99_ms{0.0};
};

struct PipelineReport {
    double elapsed_seconds{0.0};
    double total_fps{0.0};
    std::vector<StreamMetrics> streams;
    std::vector<ResultIdentity> results;
};

class MultiStreamPipeline {
public:
    MultiStreamPipeline(PipelineConfig config, std::vector<StreamConfig> streams);
    ~MultiStreamPipeline();
    PipelineReport run();
    void stop() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lesson19
