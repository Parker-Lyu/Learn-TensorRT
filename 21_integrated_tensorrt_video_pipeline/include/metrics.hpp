#pragma once

#include "config.hpp"
#include "tensorrt_backend.hpp"

#include <filesystem>
#include <map>
#include <vector>

namespace lesson21 {

struct BatchTimingSample {
    std::uint64_t batch_id{};
    std::size_t batch_size{};
    double queue_wait_ms{};
    double batch_fill_wait_ms{};
    double host_staging_ms{};
    double capacity_growth_ms{};
    double h2d_ms{};
    double gpu_preprocess_ms{};
    double tensorrt_ms{};
    double d2h_ms{};
    double cpu_postprocess_ms{};
};

class PipelineMetrics {
public:
    void record_batch(const GpuBatchResult& result, double queue_wait_ms,
                      double batch_fill_wait_ms, double cpu_postprocess_ms,
                      const std::vector<double>& frame_latencies_ms);
    void write(const std::filesystem::path& path, const PipelineConfig& config,
               const RuntimeIdentity& identity, std::uint64_t captured,
               std::uint64_t evicted, std::uint64_t aborted,
               std::uint64_t rejected_on_admission, std::size_t queue_peak,
               double elapsed_seconds) const;

    std::uint64_t submitted() const noexcept { return submitted_; }
    std::uint64_t completed() const noexcept { return completed_; }

private:
    std::uint64_t submitted_{0};
    std::uint64_t completed_{0};
    std::map<std::size_t, std::uint64_t> batch_distribution_;
    std::map<std::size_t, std::uint64_t> per_stream_completed_;
    std::map<std::size_t, std::vector<double>> per_stream_latencies_;
    std::vector<double> latencies_;
    std::vector<BatchTimingSample> batches_;
};

}  // namespace lesson21
