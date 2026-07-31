#include "frame_scheduler.hpp"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace lesson21 {
namespace {

std::vector<std::unique_ptr<FrameSource>> image_sources(
        std::vector<std::vector<cv::Mat>> images, std::size_t frame_count) {
    std::vector<std::unique_ptr<FrameSource>> result;
    result.reserve(images.size());
    for (std::size_t index = 0; index < images.size(); ++index) {
        result.push_back(make_image_sequence_source(
            std::move(images[index]), frame_count, "memory-source-" + std::to_string(index)));
    }
    return result;
}

}  // namespace

FrameScheduler::FrameScheduler(
        std::vector<cv::Mat> sources, std::size_t count, std::size_t capacity,
        OverloadPolicy policy, std::chrono::milliseconds interval, SchedulingPolicy scheduling)
    : FrameScheduler([&sources] {
          std::vector<std::vector<cv::Mat>> nested;
          nested.reserve(sources.size());
          for (cv::Mat& image : sources) nested.push_back({std::move(image)});
          return nested;
      }(), count, capacity, policy, interval, scheduling) {}

FrameScheduler::FrameScheduler(
        std::vector<std::vector<cv::Mat>> sources, std::size_t count, std::size_t capacity,
        OverloadPolicy policy, std::chrono::milliseconds interval, SchedulingPolicy scheduling)
    : FrameScheduler(image_sources(std::move(sources), count), capacity, policy,
                     interval, scheduling) {}

FrameScheduler::FrameScheduler(
        std::vector<std::unique_ptr<FrameSource>> sources, std::size_t capacity,
        OverloadPolicy policy, std::chrono::milliseconds interval, SchedulingPolicy scheduling)
    : sources_(std::move(sources)), interval_(interval), scheduling_(scheduling) {
    if (sources_.empty() || std::any_of(sources_.begin(), sources_.end(),
                                        [](const auto& source) { return source == nullptr; })) {
        throw std::invalid_argument("sources must be non-empty");
    }
    queues_.reserve(sources_.size());
    for (std::size_t index = 0; index < sources_.size(); ++index) {
        queues_.push_back(std::make_unique<BoundedQueue<ScheduledFrame>>(capacity, policy));
    }
}

FrameScheduler::~FrameScheduler() { stop(true); }

void FrameScheduler::start() {
    if (!workers_.empty()) throw std::logic_error("scheduler already started");
    stopping_ = false;
    for (std::size_t stream = 0; stream < sources_.size(); ++stream) {
        workers_.emplace_back([this, stream] { capture(stream); });
    }
}

void FrameScheduler::capture(std::size_t stream) {
    try {
        std::uint64_t frame_id = 0;
        cv::Mat image;
        while (!stopping_) {
            if (const char* failure = std::getenv("LESSON21_FAIL_SOURCE_FRAME");
                failure != nullptr && frame_id == std::stoull(failure)) {
                throw std::runtime_error("injected source read failure");
            }
            if (!sources_[stream]->read(image)) break;
            if (image.empty()) throw std::runtime_error("source returned an empty frame");
            ScheduledFrame item{image, {stream, frame_id++, 0, Clock::now(), {}}};
            ++captured_;
            if (!queues_[stream]->push(std::move(item))) break;
            if (interval_.count() != 0) std::this_thread::sleep_for(interval_);
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(error_mutex_);
            if (!source_error_) source_error_ = std::current_exception();
        }
        stopping_ = true;
        for (auto& queue : queues_) queue->close(true);
    }
    queues_[stream]->close(false);
    ++finished_;
}

std::vector<ScheduledFrame> FrameScheduler::next_batch(
        std::size_t maximum, std::chrono::milliseconds timeout) {
    if (maximum == 0) throw std::invalid_argument("batch maximum must be positive");
    rethrow_source_error();
    std::vector<ScheduledFrame> batch;
    const auto deadline = Clock::now() + timeout;
    while (batch.size() < maximum) {
        bool found = false;
        for (std::size_t checked = 0; checked < queues_.size(); ++checked) {
            const std::size_t index = (cursor_ + checked) % queues_.size();
            auto item = queues_[index]->try_pop();
            if (!item) continue;
            if (scheduling_ == SchedulingPolicy::LatestFirst) {
                while (auto newer = queues_[index]->try_pop()) {
                    item = std::move(newer);
                    ++stale_;
                }
            }
            item->metadata.batch_index = batch.size();
            batch.push_back(std::move(*item));
            cursor_ = (index + 1) % queues_.size();
            found = true;
            break;
        }
        if (!found) {
            rethrow_source_error();
            if (done() || Clock::now() >= deadline) break;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    return batch;
}

void FrameScheduler::stop(bool discard) noexcept {
    stopping_ = true;
    for (auto& queue : queues_) queue->close(discard);
    for (auto& worker : workers_) {
        if (worker.joinable()) worker.join();
    }
    workers_.clear();
}

void FrameScheduler::rethrow_source_error() const {
    std::lock_guard<std::mutex> lock(error_mutex_);
    if (source_error_) std::rethrow_exception(source_error_);
}

std::size_t FrameScheduler::evicted() const {
    std::size_t result = stale_;
    for (const auto& queue : queues_) result += queue->evicted();
    return result;
}

std::size_t FrameScheduler::discarded() const {
    std::size_t result = 0;
    for (const auto& queue : queues_) result += queue->discarded();
    return result;
}

std::size_t FrameScheduler::queue_peak() const {
    std::size_t result = 0;
    for (const auto& queue : queues_) result = std::max(result, queue->peak());
    return result;
}

std::size_t FrameScheduler::queue_depth() const {
    std::size_t result = 0;
    for (const auto& queue : queues_) result += queue->size();
    return result;
}

bool FrameScheduler::done() const {
    return finished_ == sources_.size() &&
        std::all_of(queues_.begin(), queues_.end(),
                    [](const auto& queue) { return queue->empty(); });
}

}  // namespace lesson21
