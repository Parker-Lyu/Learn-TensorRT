#pragma once

#include "frame_source.hpp"
#include "pipeline_core.hpp"

#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace lesson21 {

enum class SchedulingPolicy { RoundRobin, LatestFirst };
struct ScheduledFrame { cv::Mat image; FrameMetadata metadata; };

class FrameScheduler {
public:
    FrameScheduler(std::vector<cv::Mat> sources, std::size_t frames_per_source,
                   std::size_t queue_capacity, OverloadPolicy policy,
                   std::chrono::milliseconds interval = std::chrono::milliseconds{0},
                   SchedulingPolicy scheduling = SchedulingPolicy::RoundRobin);
    FrameScheduler(std::vector<std::vector<cv::Mat>> sources, std::size_t frames_per_source,
                   std::size_t queue_capacity, OverloadPolicy policy,
                   std::chrono::milliseconds interval = std::chrono::milliseconds{0},
                   SchedulingPolicy scheduling = SchedulingPolicy::RoundRobin);
    FrameScheduler(std::vector<std::unique_ptr<FrameSource>> sources,
                   std::size_t queue_capacity, OverloadPolicy policy,
                   std::chrono::milliseconds interval = std::chrono::milliseconds{0},
                   SchedulingPolicy scheduling = SchedulingPolicy::RoundRobin);
    ~FrameScheduler();

    void start();
    std::vector<ScheduledFrame> next_batch(std::size_t maximum,
                                           std::chrono::milliseconds timeout);
    void stop(bool discard) noexcept;
    void rethrow_source_error() const;

    std::size_t captured() const noexcept { return captured_; }
    std::size_t evicted() const;
    std::size_t discarded() const;
    std::size_t queue_peak() const;
    std::size_t queue_depth() const;
    bool done() const;

private:
    void capture(std::size_t stream);

    std::vector<std::unique_ptr<FrameSource>> sources_;
    std::chrono::milliseconds interval_;
    std::vector<std::unique_ptr<BoundedQueue<ScheduledFrame>>> queues_;
    std::vector<std::thread> workers_;
    std::atomic<std::size_t> captured_{0};
    std::atomic<std::size_t> finished_{0};
    std::atomic<std::size_t> stale_{0};
    std::atomic<bool> stopping_{false};
    std::size_t cursor_{0};
    SchedulingPolicy scheduling_;
    mutable std::mutex error_mutex_;
    std::exception_ptr source_error_;
};

}  // namespace lesson21
