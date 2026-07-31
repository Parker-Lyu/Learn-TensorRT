#pragma once
#include "pipeline_core.hpp"
#include <opencv2/core.hpp>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>
namespace lesson21 {
struct ScheduledFrame { cv::Mat image; FrameMetadata metadata; };
class FrameScheduler {
public:
 FrameScheduler(std::vector<cv::Mat> sources,std::size_t frames_per_source,std::size_t queue_capacity,OverloadPolicy policy,std::chrono::milliseconds interval=std::chrono::milliseconds{0});
 ~FrameScheduler();
 void start();
 std::vector<ScheduledFrame> next_batch(std::size_t maximum,std::chrono::milliseconds timeout);
 void stop(bool discard) noexcept;
 std::size_t captured() const noexcept{return captured_;}
 std::size_t evicted() const;
 std::size_t queue_peak() const;
 bool done() const;
private:
 std::vector<cv::Mat> sources_;std::size_t frames_per_source_;std::chrono::milliseconds interval_;
 std::vector<std::unique_ptr<BoundedQueue<ScheduledFrame>>> queues_;std::vector<std::thread> workers_;
 std::atomic<std::size_t> captured_{0},finished_{0};std::atomic<bool> stopping_{false};std::size_t cursor_{0};
};
}
