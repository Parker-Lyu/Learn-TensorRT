#include "multistream_pipeline.hpp"

#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace lesson16 {
namespace {

struct Frame {
    ResultIdentity identity;
    cv::Mat image;
    std::chrono::steady_clock::time_point captured_at;
};

class FrameQueue {
public:
    explicit FrameQueue(std::size_t capacity) : capacity_(capacity) {
        if (!capacity) throw std::invalid_argument("queue capacity must be positive");
    }
    bool push(Frame frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (closed_) return false;
        if (frames_.size() == capacity_) { frames_.pop_front(); ++dropped_; }
        frames_.push_back(std::move(frame));
        high_watermark_ = std::max(high_watermark_, frames_.size());
        return true;
    }
    std::optional<Frame> pop(bool latest) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (frames_.empty()) return std::nullopt;
        Frame frame = latest ? std::move(frames_.back()) : std::move(frames_.front());
        if (latest) {
            dropped_ += frames_.size() - 1;
            frames_.clear();
        } else frames_.pop_front();
        return frame;
    }
    void close(bool discard) {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        if (discard) { dropped_ += frames_.size(); frames_.clear(); }
    }
    bool done() const { std::lock_guard<std::mutex> lock(mutex_); return closed_ && frames_.empty(); }
    std::size_t dropped() const { std::lock_guard<std::mutex> lock(mutex_); return dropped_; }
    std::size_t peak() const { std::lock_guard<std::mutex> lock(mutex_); return high_watermark_; }
private:
    std::size_t capacity_;
    mutable std::mutex mutex_;
    std::deque<Frame> frames_;
    std::size_t dropped_{0};
    std::size_t high_watermark_{0};
    bool closed_{false};
};

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size())) - 1.0);
    return values[std::min(index, values.size() - 1)];
}

}  // namespace

struct MultiStreamPipeline::Impl {
    struct State {
        State(StreamConfig value, std::size_t capacity)
            : config(std::move(value)), queue(capacity) {}
        StreamConfig config;
        FrameQueue queue;
        std::atomic<std::size_t> captured{0};
        std::atomic<std::size_t> processed{0};
        std::atomic<std::size_t> source_failures{0};
        std::mutex latency_mutex;
        std::vector<double> latencies;
    };

    Impl(PipelineConfig value, std::vector<StreamConfig> stream_configs) : config(value) {
        if (stream_configs.size() < 2) throw std::invalid_argument("at least two streams are required");
        if (!config.max_batch_size || !config.max_in_flight_batches)
            throw std::invalid_argument("batch sizes and in-flight limit must be positive");
        std::unordered_set<std::size_t> ids;
        for (auto& stream : stream_configs) {
            if (!ids.insert(stream.stream_id).second) throw std::invalid_argument("stream IDs must be unique");
            states.push_back(std::make_unique<State>(std::move(stream), config.queue_capacity));
        }
    }

    void capture(State& state) {
        try {
            cv::VideoCapture video;
            if (!state.config.uri.empty() && !video.open(state.config.uri))
                throw std::runtime_error("failed to open stream " + std::to_string(state.config.stream_id));
            for (std::size_t frame_id = 1; !cancelled.load(); ++frame_id) {
                if (state.config.fail_at_frame && frame_id == state.config.fail_at_frame)
                    throw std::runtime_error("injected source failure for stream " +
                                             std::to_string(state.config.stream_id));
                cv::Mat image;
                if (state.config.uri.empty()) {
                    if (frame_id > state.config.synthetic_frames) break;
                    image = cv::Mat(24, 32, CV_8UC3,
                                    cv::Scalar(state.config.stream_id % 255, frame_id % 255, 64));
                } else if (!video.read(image)) break;
                ++state.captured;
                if (!state.queue.push(Frame{{state.config.stream_id, frame_id}, std::move(image),
                                            std::chrono::steady_clock::now()})) break;
                wake.notify_one();
                if (state.config.frame_interval.count()) std::this_thread::sleep_for(state.config.frame_interval);
            }
            state.queue.close(false);
            wake.notify_all();
        } catch (...) {
            ++state.source_failures;
            state.queue.close(true);
            wake.notify_all();
            if (config.source_failure == SourceFailurePolicy::StopAll) fail(std::current_exception());
        }
    }

    bool all_done() const {
        return std::all_of(states.begin(), states.end(), [](const auto& state) { return state->queue.done(); });
    }

    void dispatch(std::vector<Frame> frames) {
        const auto now = std::chrono::steady_clock::now();
        for (const auto& frame : frames) {
            auto found = std::find_if(states.begin(), states.end(), [&](const auto& state) {
                return state->config.stream_id == frame.identity.stream_id;
            });
            if (found == states.end()) throw std::logic_error("result has an unknown stream ID");
            State& state = **found;
            {
                std::lock_guard<std::mutex> lock(state.latency_mutex);
                state.latencies.push_back(
                    std::chrono::duration<double, std::milli>(now - frame.captured_at).count());
            }
            ++state.processed;
            std::lock_guard<std::mutex> lock(results_mutex);
            results.push_back(frame.identity);
        }
    }

    void schedule() {
        using BatchFuture = std::future<std::vector<Frame>>;
        std::vector<BatchFuture> pending;
        std::size_t cursor = 0;
        std::size_t batch_number = 0;
        auto collect_one = [&](bool wait) {
            while (true) {
                for (auto iterator = pending.begin(); iterator != pending.end(); ++iterator) {
                    if (iterator->wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                        dispatch(iterator->get());
                        pending.erase(iterator);
                        return;
                    }
                }
                if (!wait) return;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        };

        while (!cancelled.load()) {
            std::vector<Frame> batch;
            const auto deadline = std::chrono::steady_clock::now() + config.batch_timeout;
            while (batch.size() < config.max_batch_size) {
                bool found = false;
                for (std::size_t checked = 0; checked < states.size(); ++checked) {
                    const std::size_t index = (cursor + checked) % states.size();
                    auto frame = states[index]->queue.pop(
                        config.scheduling == SchedulingPolicy::LatestFirst);
                    if (frame) {
                        batch.push_back(std::move(*frame));
                        cursor = (index + 1) % states.size();
                        found = true;
                        break;
                    }
                }
                if (batch.size() == config.max_batch_size || all_done()) break;
                if (!found) {
                    const auto now = std::chrono::steady_clock::now();
                    if (now >= deadline) break;
                    std::unique_lock<std::mutex> lock(wake_mutex);
                    wake.wait_until(lock, deadline);
                }
            }
            if (batch.empty()) {
                if (all_done()) break;
                continue;
            }
            ++batch_number;
            if (config.fail_inference_batch && batch_number == config.fail_inference_batch)
                throw std::runtime_error("injected inference failure at batch " +
                                         std::to_string(batch_number));
            const auto delay = config.simulated_inference +
                               std::chrono::milliseconds(batch_number % 2 ? 2 : 0);
            pending.push_back(std::async(std::launch::async,
                [delay, batch = std::move(batch)]() mutable {
                    std::this_thread::sleep_for(delay);
                    return batch;
                }));
            if (pending.size() >= config.max_in_flight_batches) collect_one(true);
            else collect_one(false);
        }
        while (!pending.empty()) collect_one(true);
    }

    void stop() noexcept {
        cancelled.store(true);
        for (auto& state : states) state->queue.close(true);
        wake.notify_all();
    }
    void fail(std::exception_ptr value) noexcept {
        { std::lock_guard<std::mutex> lock(failure_mutex); if (!failure) failure = std::move(value); }
        stop();
    }

    PipelineConfig config;
    std::vector<std::unique_ptr<State>> states;
    std::atomic<bool> cancelled{false};
    std::condition_variable wake;
    std::mutex wake_mutex;
    std::mutex results_mutex;
    std::vector<ResultIdentity> results;
    std::mutex failure_mutex;
    std::exception_ptr failure;
    bool ran{false};
};

MultiStreamPipeline::MultiStreamPipeline(PipelineConfig config, std::vector<StreamConfig> streams)
    : impl_(std::make_unique<Impl>(config, std::move(streams))) {}
MultiStreamPipeline::~MultiStreamPipeline() { stop(); }

PipelineReport MultiStreamPipeline::run() {
    if (impl_->ran) throw std::logic_error("pipeline instances are single-use");
    impl_->ran = true;
    const auto start = std::chrono::steady_clock::now();
    std::vector<std::thread> captures;
    for (auto& state : impl_->states) captures.emplace_back([&, state = state.get()] { impl_->capture(*state); });
    std::thread scheduler([&] { try { impl_->schedule(); } catch (...) { impl_->fail(std::current_exception()); } });
    for (auto& capture : captures) capture.join();
    scheduler.join();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    { std::lock_guard<std::mutex> lock(impl_->failure_mutex); if (impl_->failure) std::rethrow_exception(impl_->failure); }

    PipelineReport report;
    report.elapsed_seconds = elapsed;
    report.results = impl_->results;
    std::size_t total = 0;
    for (auto& state : impl_->states) {
        StreamMetrics metrics;
        metrics.stream_id = state->config.stream_id;
        metrics.captured = state->captured.load();
        metrics.processed = state->processed.load();
        metrics.dropped = state->queue.dropped();
        metrics.source_failures = state->source_failures.load();
        metrics.queue_high_watermark = state->queue.peak();
        metrics.fps = metrics.processed / elapsed;
        metrics.latency_p50_ms = percentile(state->latencies, 0.50);
        metrics.latency_p90_ms = percentile(state->latencies, 0.90);
        metrics.latency_p99_ms = percentile(state->latencies, 0.99);
        total += metrics.processed;
        report.streams.push_back(metrics);
    }
    report.total_fps = total / elapsed;
    return report;
}
void MultiStreamPipeline::stop() noexcept { if (impl_) impl_->stop(); }

}  // namespace lesson16
