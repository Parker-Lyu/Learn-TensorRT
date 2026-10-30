#pragma once

#include "config.hpp"
#include "tensorrt_backend.hpp"

#include <filesystem>
#include <fstream>
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
    double result_output_ms{};
};

class PipelineMetrics {
public:
    explicit PipelineMetrics(const std::filesystem::path& output_directory);
    void record_submission(std::size_t frame_count);
    void record_batch(const GpuBatchResult& result, double queue_wait_ms,
                      double batch_fill_wait_ms, double cpu_postprocess_ms,
                      double result_output_ms,
                      const std::vector<double>& frame_latencies_ms);
    void write(const std::filesystem::path& path, const PipelineConfig& config,
               const RuntimeIdentity& identity, const Accounting& accounting,
               std::size_t queue_peak, double elapsed_seconds, bool normal_drain) const;

    std::uint64_t submitted() const noexcept { return submitted_; }
    std::uint64_t completed() const noexcept { return completed_; }

private:
    static constexpr std::size_t kLatencyReservoirSize = 8192;
    static constexpr std::size_t kRetainedBatchSamples = 256;
    static void add_bounded(std::vector<double>& values, double value,
                            std::uint64_t observation_count);
    void write_raw_sample(const BatchTimingSample& sample);

    std::uint64_t submitted_{0};
    std::uint64_t completed_{0};
    std::uint64_t latency_observations_{0};
    double host_staging_total_ms_{0.0};
    double h2d_total_ms_{0.0};
    double preprocess_total_ms_{0.0};
    double inference_total_ms_{0.0};
    double d2h_total_ms_{0.0};
    double postprocess_total_ms_{0.0};
    double output_total_ms_{0.0};
    std::map<std::size_t, std::uint64_t> batch_distribution_;
    std::map<std::size_t, std::uint64_t> per_stream_completed_;
    std::map<std::size_t, std::vector<double>> per_stream_latencies_;
    std::vector<double> latencies_;
    std::vector<BatchTimingSample> batches_;
    std::ofstream raw_batch_samples_;
};

}  // namespace lesson21
