#include "bounded_queue.hpp"
#include "image_pipeline.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws(Function&& function, const std::string& expected_text) {
    try {
        function();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(expected_text) != std::string::npos,
                "exception did not contain: " + expected_text);
        return;
    }
    throw std::runtime_error("expected exception containing: " + expected_text);
}

void test_invalid_capacity() {
    require_throws([] { lesson13::BoundedQueue<int> queue(0); }, "capacity");
}

void test_fifo_and_drain_close() {
    lesson13::BoundedQueue<int> queue(2);
    require(queue.push(1) == lesson13::PushResult::Pushed, "first push failed");
    require(queue.push(2) == lesson13::PushResult::Pushed, "second push failed");
    queue.close(lesson13::CloseMode::Drain);
    require(queue.push(3) == lesson13::PushResult::Closed, "closed queue accepted a push");
    require(queue.pop() == 1, "FIFO first value mismatch");
    require(queue.pop() == 2, "FIFO second value mismatch");
    require(!queue.pop().has_value(), "drained queue should return nullopt");
}

void test_drop_policies() {
    lesson13::BoundedQueue<int> newest(1, lesson13::OverloadPolicy::DropNewest);
    require(newest.push(1) == lesson13::PushResult::Pushed, "initial newest push failed");
    require(newest.push(2) == lesson13::PushResult::DroppedNewest,
            "drop-newest result mismatch");
    newest.close();
    require(newest.pop() == 1, "drop-newest kept the wrong value");

    lesson13::BoundedQueue<int> oldest(1, lesson13::OverloadPolicy::DropOldest);
    require(oldest.push(1) == lesson13::PushResult::Pushed, "initial oldest push failed");
    require(oldest.push(2) == lesson13::PushResult::DroppedOldest,
            "drop-oldest result mismatch");
    oldest.close();
    require(oldest.pop() == 2, "drop-oldest kept the wrong value");
    require(oldest.stats().dropped == 1, "drop count mismatch");
}

void test_close_wakes_blocked_threads() {
    using namespace std::chrono_literals;
    lesson13::BoundedQueue<int> full_queue(1, lesson13::OverloadPolicy::Block);
    full_queue.push(1);
    auto producer = std::async(std::launch::async, [&] { return full_queue.push(2); });
    require(producer.wait_for(20ms) == std::future_status::timeout,
            "producer did not block on a full queue");
    full_queue.close(lesson13::CloseMode::Discard);
    require(producer.wait_for(1s) == std::future_status::ready,
            "close did not wake blocked producer");
    require(producer.get() == lesson13::PushResult::Closed,
            "woken producer did not observe closed state");
    require(!full_queue.pop().has_value(), "discard close retained queued work");
    require(full_queue.stats().dropped == 1, "discard close did not account for queued work");

    lesson13::BoundedQueue<int> empty_queue(1);
    auto consumer = std::async(std::launch::async, [&] { return empty_queue.pop(); });
    require(consumer.wait_for(20ms) == std::future_status::timeout,
            "consumer did not block on an empty queue");
    empty_queue.close();
    require(consumer.wait_for(1s) == std::future_status::ready,
            "close did not wake blocked consumer");
    require(!consumer.get().has_value(), "closed empty queue returned a value");
}

std::filesystem::path make_fixture() {
    const auto path = std::filesystem::temp_directory_path() /
                      ("lesson13_fixture_" + std::to_string(
                          std::chrono::steady_clock::now().time_since_epoch().count()) + ".img");
    std::ofstream output(path, std::ios::binary);
    output << "small deterministic image fixture";
    if (!output) {
        throw std::runtime_error("failed to create test fixture");
    }
    return path;
}

void test_pipeline_overload_and_repeated_start_stop() {
    const auto fixture = make_fixture();
    try {
        for (int iteration = 0; iteration < 20; ++iteration) {
            lesson13::PipelineConfig config;
            config.frame_count = 30;
            config.queue_capacity = 2;
            config.overload_policy = lesson13::OverloadPolicy::DropOldest;
            config.producer_delay = std::chrono::milliseconds(0);
            config.consumer_delay = std::chrono::milliseconds(1);
            lesson13::ImagePipeline pipeline(config);
            const auto stats = pipeline.run({fixture});
            require(stats.frames_read == config.frame_count, "pipeline did not read every frame");
            require(stats.queue.high_watermark <= config.queue_capacity,
                    "queue exceeded configured capacity");
            require(stats.frames_processed + stats.queue.dropped == stats.frames_read,
                    "processed and dropped accounting mismatch");
        }
    } catch (...) {
        std::filesystem::remove(fixture);
        throw;
    }
    std::filesystem::remove(fixture);
}

void test_pipeline_failure_propagation() {
    const auto fixture = make_fixture();
    try {
        lesson13::PipelineConfig config;
        config.frame_count = 10;
        config.queue_capacity = 1;
        config.overload_policy = lesson13::OverloadPolicy::Block;
        config.fail_consumer_at = 2;
        lesson13::ImagePipeline pipeline(config);
        require_throws([&] { pipeline.run({fixture}); }, "consumer failure");

        config.fail_consumer_at = 0;
        config.fail_producer_at = 2;
        lesson13::ImagePipeline producer_failure_pipeline(config);
        require_throws([&] { producer_failure_pipeline.run({fixture}); }, "producer failure");
    } catch (...) {
        std::filesystem::remove(fixture);
        throw;
    }
    std::filesystem::remove(fixture);
}

void test_explicit_stop() {
    using namespace std::chrono_literals;
    const auto fixture = make_fixture();
    try {
        lesson13::PipelineConfig config;
        config.frame_count = 10000;
        config.queue_capacity = 2;
        config.overload_policy = lesson13::OverloadPolicy::Block;
        config.producer_delay = 0ms;
        config.consumer_delay = 2ms;
        lesson13::ImagePipeline pipeline(config);
        auto run = std::async(std::launch::async, [&] { return pipeline.run({fixture}); });
        std::this_thread::sleep_for(10ms);
        pipeline.stop();
        require(run.wait_for(1s) == std::future_status::ready,
                "explicit stop did not terminate the pipeline");
        require(run.get().queue.high_watermark <= config.queue_capacity,
                "stopped pipeline exceeded queue capacity");
    } catch (...) {
        std::filesystem::remove(fixture);
        throw;
    }
    std::filesystem::remove(fixture);
}

}  // namespace

int main() {
    try {
        test_invalid_capacity();
        test_fifo_and_drain_close();
        test_drop_policies();
        test_close_wakes_blocked_threads();
        test_pipeline_overload_and_repeated_start_stop();
        test_pipeline_failure_propagation();
        test_explicit_stop();
        std::cout << "All producer-consumer tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
