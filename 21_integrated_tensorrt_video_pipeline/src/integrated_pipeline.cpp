#include "integrated_pipeline.hpp"

#include "frame_scheduler.hpp"
#include "frame_source.hpp"
#include "metrics.hpp"
#include "result_writer.hpp"
#include "tensorrt_backend.hpp"

#include <chrono>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lesson21 {
namespace {

struct PendingBatch {
    std::size_t slot{};
    std::vector<cv::Mat> images;
    double queue_wait_ms{};
    double batch_fill_wait_ms{};
};

std::vector<std::unique_ptr<FrameSource>> sources(const PipelineConfig& config) {
    const std::size_t frames_per_source = config.duration.count() == 0
        ? std::max<std::size_t>(1, config.frame_count / config.source_paths.size())
        : std::numeric_limits<std::size_t>::max();
    std::vector<std::unique_ptr<FrameSource>> result;
    result.reserve(config.source_paths.size());
    for (const std::string& path : config.source_paths) {
        result.push_back(make_path_source(path, frames_per_source, config.repeat_source));
    }
    return result;
}

}  // namespace

int run_integrated_pipeline(const PipelineConfig& config) {
    TensorRtBackend backend(config.engine_path, config.slot_count, {640, 640});
    FrameScheduler scheduler(sources(config), config.queue_capacity, config.overload_policy,
                             config.capture_interval, config.scheduling_policy);
    ResultWriter writer(config.output_directory);
    PipelineMetrics metrics;
    const RuntimeIdentity identity = backend.identity();
    std::ofstream snapshots(config.output_directory / "metrics_snapshots.jsonl");
    if (!snapshots) throw std::runtime_error("cannot open metrics snapshot output");
    std::deque<PendingBatch> pending;
    std::uint64_t next_batch = 0;
    const auto started = Clock::now();
    const auto deadline = config.duration.count() == 0
        ? Clock::time_point::max() : started + config.duration;
    auto next_snapshot = started;
    scheduler.start();
    try {
        while (!scheduler.done() || !pending.empty()) {
            if (Clock::now() >= next_snapshot) {
                snapshots << "{\"elapsed_seconds\":"
                          << std::chrono::duration<double>(Clock::now() - started).count()
                          << ",\"queue_depth\":" << scheduler.queue_depth()
                          << ",\"queue_peak\":" << scheduler.queue_peak()
                          << ",\"captured\":" << scheduler.captured()
                          << ",\"submitted\":" << metrics.submitted()
                          << ",\"completed\":" << metrics.completed()
                          << ",\"evicted\":" << scheduler.evicted()
                          << ",\"available_slots\":" << backend.available_slots()
                          << ",\"errors\":0}\n";
                snapshots.flush();
                next_snapshot = Clock::now() + config.metrics_interval;
            }
            if (Clock::now() >= deadline) scheduler.stop(true);
            while (backend.available_slots() != 0 && !scheduler.done()) {
                const auto fill_started = Clock::now();
                auto scheduled = scheduler.next_batch(config.maximum_batch,
                                                      std::chrono::milliseconds(4));
                const double batch_fill_ms = std::chrono::duration<double, std::milli>(
                    Clock::now() - fill_started).count();
                if (scheduled.empty()) break;
                const auto reserved = backend.try_reserve();
                if (!reserved) throw std::logic_error("available slot could not be reserved");
                std::vector<cv::Mat> images;
                BatchMetadata metadata;
                metadata.batch_id = next_batch++;
                double queue_wait_ms = 0.0;
                for (ScheduledFrame& frame : scheduled) {
                    queue_wait_ms += std::chrono::duration<double, std::milli>(
                        Clock::now() - frame.metadata.captured_at).count();
                    images.push_back(frame.image);
                    metadata.frames.push_back(frame.metadata);
                }
                queue_wait_ms /= static_cast<double>(scheduled.size());
                backend.submit(*reserved, images, std::move(metadata));
                pending.push_back({*reserved, std::move(images), queue_wait_ms, batch_fill_ms});
                if (const char* abort_after = std::getenv("LESSON21_ABORT_AFTER_SUBMISSIONS");
                    abort_after != nullptr && next_batch >= std::stoull(abort_after)) {
                    throw std::runtime_error("injected abort with submitted work");
                }
            }

            bool collected = false;
            for (auto iterator = pending.begin(); iterator != pending.end(); ++iterator) {
                if (!backend.ready(iterator->slot)) continue;
                PendingBatch current = std::move(*iterator);
                pending.erase(iterator);
                GpuBatchResult result = backend.collect(current.slot);
                if (const char* failure = std::getenv("LESSON21_FAIL_POSTPROCESS_BATCH");
                    failure != nullptr && result.metadata.batch_id == std::stoull(failure)) {
                    throw std::runtime_error("injected postprocess failure");
                }
                std::vector<double> frame_latencies;
                const double postprocess_ms = writer.write(result, current.images, frame_latencies);
                metrics.record_batch(result, current.queue_wait_ms, current.batch_fill_wait_ms,
                                     postprocess_ms, frame_latencies);
                collected = true;
                break;
            }
            if (!collected && !pending.empty()) {
                PendingBatch current = std::move(pending.front());
                pending.pop_front();
                GpuBatchResult result = backend.collect(current.slot);
                std::vector<double> frame_latencies;
                const double postprocess_ms = writer.write(result, current.images, frame_latencies);
                metrics.record_batch(result, current.queue_wait_ms, current.batch_fill_wait_ms,
                                     postprocess_ms, frame_latencies);
            }
        }
        scheduler.stop(false);
        scheduler.rethrow_source_error();
    } catch (...) {
        const std::exception_ptr causal = std::current_exception();
        scheduler.stop(true);
        for (const PendingBatch& batch : pending) {
            try { backend.collect(batch.slot); } catch (...) { /* Preserve the first error. */ }
        }
        std::rethrow_exception(causal);
    }

    const double elapsed = std::chrono::duration<double>(Clock::now() - started).count();
    const std::uint64_t aborted = scheduler.discarded();
    metrics.write(config.output_directory / "metrics.json", config, identity,
                  scheduler.captured(), scheduler.evicted(), aborted,
                  scheduler.rejected_on_close(), scheduler.queue_peak(), elapsed);
    const bool accounting_ok = scheduler.captured() ==
        metrics.completed() + scheduler.evicted() + aborted;
    std::cout << "backend=tensorrt captured=" << scheduler.captured()
              << " processed=" << metrics.completed()
              << " dropped=" << scheduler.evicted() + aborted
              << " queue_peak=" << scheduler.queue_peak()
              << " fps=" << (elapsed > 0.0 ? metrics.completed() / elapsed : 0.0) << '\n';
    return accounting_ok ? 0 : 1;
}

}  // namespace lesson21
