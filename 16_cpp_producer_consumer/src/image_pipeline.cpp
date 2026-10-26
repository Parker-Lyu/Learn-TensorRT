#include "image_pipeline.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>

namespace lesson16 {
namespace {

std::vector<std::uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open image: " + path.string());
    }

    std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (input.bad()) {
        throw std::runtime_error("failed while reading image: " + path.string());
    }
    if (bytes.empty()) {
        throw std::runtime_error("image file is empty: " + path.string());
    }
    return bytes;
}

}  // namespace

ImagePipeline::ImagePipeline(PipelineConfig config)
    : config_(config), queue_(config.queue_capacity, config.overload_policy) {
    if (config_.frame_count == 0) {
        throw std::invalid_argument("frame count must be greater than zero");
    }
    if (config_.producer_delay.count() < 0 || config_.consumer_delay.count() < 0) {
        throw std::invalid_argument("producer and consumer delays must not be negative");
    }
}

ImagePipeline::~ImagePipeline() {
    stop();
}

PipelineStats ImagePipeline::run(const std::vector<std::filesystem::path>& image_paths) {
    if (image_paths.empty()) {
        throw std::invalid_argument("at least one image path is required");
    }
    if (run_called_) {
        throw std::logic_error("an ImagePipeline instance can only be run once");
    }
    run_called_ = true;

    std::thread producer([this, &image_paths] {
        try {
            producer_loop(image_paths);
            queue_.close(CloseMode::Drain);
        } catch (...) {
            record_failure(std::current_exception());
        }
    });
    std::thread consumer;
    try {
        consumer = std::thread([this] {
            try {
                consumer_loop();
            } catch (...) {
                record_failure(std::current_exception());
            }
        });
    } catch (...) {
        stop();
        producer.join();
        throw;
    }

    producer.join();
    consumer.join();

    {
        std::lock_guard<std::mutex> lock(failure_mutex_);
        if (failure_) {
            std::rethrow_exception(failure_);
        }
    }

    PipelineStats stats;
    stats.frames_read = frames_read_.load();
    stats.frames_processed = frames_processed_.load();
    stats.queue = queue_.stats();
    stats.average_queue_latency_ms = average_queue_latency_ms_;
    stats.max_queue_latency_ms = max_queue_latency_ms_;
    return stats;
}

void ImagePipeline::stop() noexcept {
    {
        // Use the waiter's mutex when changing its predicate to avoid a lost wakeup.
        std::lock_guard<std::mutex> lock(delay_mutex_);
        stop_requested_.store(true);
    }
    delay_cv_.notify_all();
    queue_.close(CloseMode::Discard);
}

bool ImagePipeline::wait_or_stopped(std::chrono::milliseconds delay) {
    if (delay.count() == 0) {
        return stop_requested_.load();
    }
    std::unique_lock<std::mutex> lock(delay_mutex_);
    return delay_cv_.wait_for(lock, delay, [this] { return stop_requested_.load(); });
}

void ImagePipeline::producer_loop(const std::vector<std::filesystem::path>& image_paths) {
    for (std::size_t index = 0; index < config_.frame_count; ++index) {
        const std::size_t sequence = index + 1;
        if (stop_requested_.load()) {
            return;
        }
        if (config_.fail_producer_at != 0 && sequence == config_.fail_producer_at) {
            throw std::runtime_error("injected producer failure at frame " +
                                     std::to_string(sequence));
        }

        const auto& path = image_paths[(sequence - 1) % image_paths.size()];
        ImageFrame frame{sequence, path, read_file(path), std::chrono::steady_clock::now()};
        ++frames_read_;

        const PushResult result = queue_.push(std::move(frame));
        if (result == PushResult::Closed) {
            return;
        }
        if (wait_or_stopped(config_.producer_delay)) {
            return;
        }
    }
}

void ImagePipeline::consumer_loop() {
    while (const auto frame = queue_.pop()) {
        const std::size_t next_processed = frames_processed_.load() + 1;
        if (config_.fail_consumer_at != 0 && next_processed == config_.fail_consumer_at) {
            throw std::runtime_error("injected consumer failure after " +
                                     std::to_string(next_processed - 1) + " frames");
        }

        const auto now = std::chrono::steady_clock::now();
        const auto queued =
            std::chrono::duration<double, std::milli>(now - frame->captured_at).count();
        ++latency_samples_;
        average_queue_latency_ms_ +=
            (queued - average_queue_latency_ms_) / static_cast<double>(latency_samples_);
        max_queue_latency_ms_ = std::max(max_queue_latency_ms_, queued);

        // Interruptible simulated work, not cancellation of an actual GPU operation.
        if (wait_or_stopped(config_.consumer_delay)) {
            return;
        }
        ++frames_processed_;
    }
}

void ImagePipeline::record_failure(std::exception_ptr failure) noexcept {
    {
        std::lock_guard<std::mutex> lock(failure_mutex_);
        if (!failure_) {
            failure_ = std::move(failure);
        }
    }
    stop();
}

}  // namespace lesson16
