#include "frame_scheduler.hpp"

#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>

#define CHECK(x) do { if (!(x)) throw std::runtime_error("check failed: " #x); } while (false)

namespace {
using namespace std::chrono_literals;

class FailingSource final : public lesson21::FrameSource {
public:
    bool read(cv::Mat& frame) override {
        if (reads_++ != 0) throw std::runtime_error("expected source failure");
        frame = cv::Mat(8, 8, CV_8UC3, cv::Scalar(1, 2, 3));
        return true;
    }
    std::string name() const override { return "failing"; }
private:
    int reads_{0};
};

class PartialFailureSource final : public lesson21::FrameSource {
public:
    PartialFailureSource(std::promise<void>& queued, std::shared_future<void> fail)
        : queued_(queued), fail_(std::move(fail)) {}
    bool read(cv::Mat& frame) override {
        if (reads_++ == 0) {
            frame = cv::Mat(8, 8, CV_8UC3, cv::Scalar(1, 2, 3));
            return true;
        }
        queued_.set_value();
        fail_.wait();
        throw std::runtime_error("failure during batch assembly");
    }
    std::string name() const override { return "partial-failure"; }
private:
    std::promise<void>& queued_;
    std::shared_future<void> fail_;
    int reads_{0};
};

// Signals EOF after all frames have been pushed, without sleeps as synchronization.
class FiniteSource final : public lesson21::FrameSource {
public:
    explicit FiniteSource(std::promise<void>& eof) : eof_(eof) {}
    bool read(cv::Mat& frame) override {
        if (reads_ == 4) {
            eof_.set_value();
            return false;
        }
        // Mat::create reuses a nonempty destination, just as a decoder may do.
        frame.create(8, 8, CV_8UC3);
        frame.setTo(cv::Scalar(reads_++, 2, 3));
        return true;
    }
    std::string name() const override { return "finite"; }
private:
    std::promise<void>& eof_;
    int reads_{0};
};

void block_policy_drains_all_identities() {
    cv::Mat image(8, 8, CV_8UC3, cv::Scalar(1, 2, 3));
    lesson21::FrameScheduler scheduler(std::vector<cv::Mat>{image, image}, 4, 2,
                                      lesson21::OverloadPolicy::Block);
    scheduler.start();
    std::set<std::pair<std::size_t, std::uint64_t>> ids;
    while (!scheduler.done()) {
        auto batch = scheduler.next_batch(3, 5ms);
        CHECK(batch.size() <= 3);
        for (std::size_t index = 0; index < batch.size(); ++index) {
            CHECK(batch[index].metadata.batch_index == index);
            CHECK(ids.emplace(batch[index].metadata.stream_id,
                              batch[index].metadata.frame_id).second);
        }
    }
    scheduler.stop(false);
    CHECK(ids.size() == 8);
    CHECK(scheduler.captured() == 8);
    CHECK(scheduler.evicted() == 0);
    CHECK(scheduler.queue_peak() <= 2);
}

void queued_frames_follow_selected_policy(lesson21::SchedulingPolicy policy) {
    std::promise<void> first_eof, second_eof;
    auto first = first_eof.get_future();
    auto second = second_eof.get_future();
    std::vector<std::unique_ptr<lesson21::FrameSource>> sources;
    sources.push_back(std::make_unique<FiniteSource>(first_eof));
    sources.push_back(std::make_unique<FiniteSource>(second_eof));
    lesson21::FrameScheduler scheduler(std::move(sources), 4,
                                      lesson21::OverloadPolicy::Block, 0ms, policy);
    scheduler.start();
    const bool finished = first.wait_for(2s) == std::future_status::ready &&
                          second.wait_for(2s) == std::future_status::ready;
    scheduler.stop(false);
    CHECK(finished);
    auto batch = scheduler.next_batch(8, 0ms);
    if (policy == lesson21::SchedulingPolicy::LatestFirst) {
        CHECK(batch.size() == 2);
        CHECK(batch[0].metadata.frame_id == 3 && batch[1].metadata.frame_id == 3);
        CHECK(scheduler.evicted() == 6);
    } else {
        CHECK(batch.size() == 8);
        for (std::size_t index = 0; index < batch.size(); ++index) {
            CHECK(batch[index].metadata.stream_id == index % 2);
            CHECK(batch[index].metadata.frame_id == index / 2);
        }
        CHECK(scheduler.evicted() == 0);
    }
    for (const auto& frame : batch) {
        CHECK(frame.image.at<cv::Vec3b>(0, 0)[0] == frame.metadata.frame_id);
    }
    CHECK(scheduler.done());
    CHECK(scheduler.captured() == batch.size() + scheduler.evicted());
}

void partial_batch_is_accounted_when_source_fails() {
    std::promise<void> queued, fail;
    auto queued_future = queued.get_future();
    std::vector<std::unique_ptr<lesson21::FrameSource>> sources;
    sources.push_back(std::make_unique<PartialFailureSource>(queued, fail.get_future().share()));
    lesson21::FrameScheduler scheduler(std::move(sources), 2, lesson21::OverloadPolicy::Block);
    scheduler.start();
    const bool frame_queued = queued_future.wait_for(2s) == std::future_status::ready;
    auto consumer = std::async(std::launch::async, [&] {
        try { scheduler.next_batch(2, 5s); }
        catch (const std::runtime_error&) { return true; }
        return false;
    });
    const auto deadline = lesson21::Clock::now() + 2s;
    while (scheduler.queue_depth() != 0 && lesson21::Clock::now() < deadline) {
        std::this_thread::yield();
    }
    const bool dequeued = scheduler.queue_depth() == 0;
    fail.set_value();
    const bool observed = consumer.get();
    scheduler.stop(true);
    CHECK(frame_queued && dequeued && observed);
    CHECK(scheduler.captured() == 1 && scheduler.discarded() == 1);
}

void source_error_reaches_consumer() {
    std::vector<std::unique_ptr<lesson21::FrameSource>> sources;
    sources.push_back(std::make_unique<FailingSource>());
    lesson21::FrameScheduler scheduler(std::move(sources), 2, lesson21::OverloadPolicy::Block);
    scheduler.start();
    bool observed = false;
    const auto deadline = lesson21::Clock::now() + 2s;
    while (!observed && lesson21::Clock::now() < deadline) {
        try { scheduler.next_batch(1, 1ms); }
        catch (const std::runtime_error&) { observed = true; }
    }
    scheduler.stop(true);
    CHECK(observed);
}
}  // namespace

int main() {
    try {
        block_policy_drains_all_identities();
        queued_frames_follow_selected_policy(lesson21::SchedulingPolicy::RoundRobin);
        queued_frames_follow_selected_policy(lesson21::SchedulingPolicy::LatestFirst);
        source_error_reaches_consumer();
        partial_batch_is_accounted_when_source_fails();
        std::cout << "frame scheduler tests passed (drain, fairness, freshness, source failure)\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
