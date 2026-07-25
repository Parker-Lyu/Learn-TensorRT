#include "async_pipeline.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <future>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace lesson15 {

class AsyncVideoPipeline::FrameQueue {
public:
    explicit FrameQueue(std::size_t capacity) : capacity_(capacity) {
        if (capacity == 0) throw std::invalid_argument("queue capacity must be positive");
    }

    bool push(Frame frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return false;
        if (frames_.size() == capacity_) {
            frames_.pop_front();
            ++dropped_;
        }
        frames_.push_back(std::move(frame));
        high_watermark_ = std::max(high_watermark_, frames_.size());
        ready_.notify_one();
        return true;
    }

    std::optional<Frame> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [&] { return closed_ || !frames_.empty(); });
        return pop_locked();
    }

    std::optional<Frame> pop_for(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait_for(lock, timeout, [&] { return closed_ || !frames_.empty(); });
        return pop_locked();
    }

    void close(bool discard) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return;
        closed_ = true;
        if (discard) {
            dropped_ += frames_.size();
            frames_.clear();
        }
        ready_.notify_all();
    }

    std::size_t dropped() const { std::lock_guard<std::mutex> lock(mutex_); return dropped_; }
    std::size_t high_watermark() const {
        std::lock_guard<std::mutex> lock(mutex_); return high_watermark_;
    }

private:
    std::optional<Frame> pop_locked() {
        if (frames_.empty()) return std::nullopt;
        Frame frame = std::move(frames_.front());
        frames_.pop_front();
        return frame;
    }
    const std::size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Frame> frames_;
    std::size_t dropped_{0};
    std::size_t high_watermark_{0};
    bool closed_{false};
};

struct VideoSource::Impl {
    explicit Impl(const std::string& uri) {
        const bool numeric = !uri.empty() &&
            std::all_of(uri.begin(), uri.end(), [](unsigned char value) {
                return std::isdigit(value) != 0;
            });
        if (numeric) capture.open(std::stoi(uri)); else capture.open(uri);
        if (!capture.isOpened()) throw std::runtime_error("failed to open video or camera: " + uri);
    }
    cv::VideoCapture capture;
};

VideoSource::VideoSource(const std::string& uri) : impl_(std::make_unique<Impl>(uri)) {}
VideoSource::~VideoSource() = default;
std::optional<cv::Mat> VideoSource::read() {
    cv::Mat frame;
    if (!impl_->capture.read(frame)) return std::nullopt;
    if (frame.empty()) throw std::runtime_error("video source returned an empty frame");
    return frame;
}

SyntheticSource::SyntheticSource(std::size_t frame_count, cv::Size size)
    : remaining_(frame_count), size_(size) {
    if (frame_count == 0 || size.width <= 0 || size.height <= 0)
        throw std::invalid_argument("synthetic source requires positive dimensions and frame count");
}
std::optional<cv::Mat> SyntheticSource::read() {
    if (remaining_ == 0) return std::nullopt;
    --remaining_;
    ++sequence_;
    cv::Mat image(size_, CV_8UC3, cv::Scalar(sequence_ % 255, 64, 128));
    return image;
}

AsyncVideoPipeline::AsyncVideoPipeline(PipelineConfig config,
                                       std::unique_ptr<FrameSource> source)
    : config_(config), source_(std::move(source)),
      queue_(std::make_unique<FrameQueue>(config.queue_capacity)) {
    if (!source_) throw std::invalid_argument("frame source must not be null");
    if (config_.max_batch_size == 0) throw std::invalid_argument("max batch size must be positive");
}
AsyncVideoPipeline::~AsyncVideoPipeline() { stop(); }

PipelineMetrics AsyncVideoPipeline::run() {
    if (ran_) throw std::logic_error("pipeline instances are single-use");
    ran_ = true;
    const auto start = std::chrono::steady_clock::now();
    std::thread capture([&] { try { capture_loop(); } catch (...) { fail(std::current_exception()); } });
    std::thread inference([&] { try { inference_loop(); } catch (...) { fail(std::current_exception()); } });
    capture.join();
    inference.join();
    const auto stop_time = std::chrono::steady_clock::now();

    { std::lock_guard<std::mutex> lock(failure_mutex_); if (failure_) std::rethrow_exception(failure_); }
    std::sort(latencies_ms_.begin(), latencies_ms_.end());
    auto percentile = [&](double fraction) {
        if (latencies_ms_.empty()) return 0.0;
        const auto index = static_cast<std::size_t>(
            std::ceil(fraction * static_cast<double>(latencies_ms_.size())) - 1.0);
        return latencies_ms_[std::min(index, latencies_ms_.size() - 1)];
    };
    PipelineMetrics metrics;
    metrics.captured = captured_.load();
    metrics.processed = processed_.load();
    metrics.dropped = queue_->dropped();
    metrics.queue_high_watermark = queue_->high_watermark();
    metrics.elapsed_seconds = std::chrono::duration<double>(stop_time - start).count();
    metrics.fps = metrics.processed / metrics.elapsed_seconds;
    metrics.latency_p50_ms = percentile(0.50);
    metrics.latency_p90_ms = percentile(0.90);
    metrics.latency_p99_ms = percentile(0.99);
    return metrics;
}

void AsyncVideoPipeline::capture_loop() {
    for (std::size_t id = 1; !cancelled_.load(); ++id) {
        if (config_.fail_capture_at && id == config_.fail_capture_at)
            throw std::runtime_error("injected capture failure at frame " + std::to_string(id));
        auto image = source_->read();
        if (!image) break;
        ++captured_;
        if (!queue_->push(Frame{id, std::move(*image), std::chrono::steady_clock::now()})) break;
    }
    queue_->close(false);
}

void AsyncVideoPipeline::inference_loop() {
    using Batch = std::vector<Frame>;
    std::deque<std::future<Batch>> in_flight;
    auto collect = [&] {
        Batch completed = in_flight.front().get();
        in_flight.pop_front();
        const auto completed_at = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(result_mutex_);
        for (const auto& frame : completed) {
            latencies_ms_.push_back(
                std::chrono::duration<double, std::milli>(completed_at - frame.captured_at).count());
            ++processed_;
        }
    };

    std::size_t submitted = 0;
    while (!cancelled_.load()) {
        auto first = queue_->pop();
        if (!first) break;
        Batch batch;
        batch.push_back(std::move(*first));
        const auto deadline = std::chrono::steady_clock::now() + config_.batch_timeout;
        while (batch.size() < config_.max_batch_size) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) break;
            auto next = queue_->pop_for(
                std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now));
            if (!next) break;
            batch.push_back(std::move(*next));
        }
        submitted += batch.size();
        if (config_.fail_worker_at && submitted >= config_.fail_worker_at)
            throw std::runtime_error("injected inference worker failure");

        const auto delay = config_.simulated_inference;
        in_flight.push_back(std::async(std::launch::async, [delay, batch = std::move(batch)]() mutable {
            // Resize stands in for CPU preprocessing; the async task represents one GPU stream slot.
            for (auto& frame : batch) cv::resize(frame.image, frame.image, {640, 640});
            std::this_thread::sleep_for(delay);
            return batch;
        }));
        // Two in-flight batches are the double buffer: one may execute while the next is prepared.
        if (in_flight.size() == 2) collect();
    }
    while (!in_flight.empty()) collect();
}

void AsyncVideoPipeline::stop() noexcept {
    cancelled_.store(true);
    if (queue_) queue_->close(true);
}
void AsyncVideoPipeline::fail(std::exception_ptr error) noexcept {
    { std::lock_guard<std::mutex> lock(failure_mutex_); if (!failure_) failure_ = std::move(error); }
    cancelled_.store(true);
    queue_->close(true);
}

}  // namespace lesson15
