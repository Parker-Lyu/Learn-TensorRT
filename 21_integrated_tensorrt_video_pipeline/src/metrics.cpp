#include "metrics.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace lesson21 {
namespace {

double percentile(std::vector<double> values, double quantile) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return values[static_cast<std::size_t>((values.size() - 1) * quantile)];
}

}  // namespace

PipelineMetrics::PipelineMetrics(const std::filesystem::path& output_directory) {
    std::filesystem::create_directories(output_directory);
    raw_batch_samples_.open(output_directory / "batch_timing_samples.jsonl");
    if (!raw_batch_samples_) throw std::runtime_error("cannot open raw batch timing output");
}

void PipelineMetrics::add_bounded(std::vector<double>& values, double value,
                                  std::uint64_t observation_count) {
    if (values.size() < kLatencyReservoirSize) {
        values.push_back(value);
        return;
    }
    // A deterministic rolling reservoir bounds service memory while retaining recent steady-state
    // behavior. Exact per-batch evidence is streamed separately to JSONL.
    values[static_cast<std::size_t>(observation_count % kLatencyReservoirSize)] = value;
}

void PipelineMetrics::write_raw_sample(const BatchTimingSample& sample) {
    raw_batch_samples_ << "{\"batch_id\":" << sample.batch_id
        << ",\"batch_size\":" << sample.batch_size
        << ",\"queue_wait_ms\":" << sample.queue_wait_ms
        << ",\"batch_fill_wait_ms\":" << sample.batch_fill_wait_ms
        << ",\"host_staging_ms\":" << sample.host_staging_ms
        << ",\"capacity_growth_ms\":" << sample.capacity_growth_ms
        << ",\"h2d_ms\":" << sample.h2d_ms
        << ",\"gpu_preprocess_ms\":" << sample.gpu_preprocess_ms
        << ",\"tensorrt_ms\":" << sample.tensorrt_ms
        << ",\"d2h_ms\":" << sample.d2h_ms
        << ",\"cpu_postprocess_ms\":" << sample.cpu_postprocess_ms << "}\n";
}

void PipelineMetrics::record_batch(const GpuBatchResult& result, double queue_wait_ms,
                                   double batch_fill_wait_ms, double cpu_postprocess_ms,
                                   const std::vector<double>& frame_latencies_ms) {
    const std::size_t batch_size = result.metadata.frames.size();
    submitted_ += batch_size;
    completed_ += batch_size;
    host_staging_total_ms_ += result.host_staging_ms;
    h2d_total_ms_ += result.h2d_ms;
    preprocess_total_ms_ += result.preprocess_ms;
    inference_total_ms_ += result.inference_ms;
    d2h_total_ms_ += result.d2h_ms;
    postprocess_total_ms_ += cpu_postprocess_ms;
    ++batch_distribution_[batch_size];
    const BatchTimingSample sample{
        result.metadata.batch_id, batch_size, queue_wait_ms, batch_fill_wait_ms,
        result.host_staging_ms, result.capacity_growth_ms, result.h2d_ms,
        result.preprocess_ms, result.inference_ms, result.d2h_ms, cpu_postprocess_ms};
    write_raw_sample(sample);
    if (batches_.size() < kRetainedBatchSamples) batches_.push_back(sample);
    else batches_[static_cast<std::size_t>(result.metadata.batch_id % kRetainedBatchSamples)] = sample;
    for (std::size_t index = 0; index < batch_size; ++index) {
        const std::size_t stream = result.metadata.frames[index].stream_id;
        const std::uint64_t stream_count = ++per_stream_completed_[stream];
        if (index < frame_latencies_ms.size()) {
            add_bounded(latencies_, frame_latencies_ms[index], ++latency_observations_);
            add_bounded(per_stream_latencies_[stream], frame_latencies_ms[index], stream_count);
        }
    }
}

void PipelineMetrics::write(const std::filesystem::path& path, const PipelineConfig& config,
                            const RuntimeIdentity& identity, std::uint64_t captured,
                            std::uint64_t evicted, std::uint64_t aborted,
                            std::uint64_t rejected_on_admission,
                            std::size_t queue_peak, double elapsed_seconds) const {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("cannot write metrics: " + path.string());
    const std::uint64_t failed = captured >= completed_ + evicted + aborted
        ? captured - completed_ - evicted - aborted : 0;
    std::uint64_t total_batches = 0;
    for (const auto& [size, count] : batch_distribution_) {
        static_cast<void>(size);
        total_batches += count;
    }
    output << std::fixed << std::setprecision(6);
    output << "{\"schema_version\":2,\"clock_domains\":{"
           << "\"host\":\"std::chrono::steady_clock\","
           << "\"gpu\":\"CUDA events on each slot stream\"},"
           << "\"captured\":" << captured
           << ",\"admitted\":" << (captured - rejected_on_admission)
           << ",\"rejected_on_admission\":" << rejected_on_admission
           << ",\"submitted\":" << submitted_
           << ",\"completed\":" << completed_
           << ",\"processed\":" << completed_
           << ",\"evicted\":" << evicted
           << ",\"dropped\":" << (evicted + aborted + failed)
           << ",\"failed\":" << failed
           << ",\"aborted\":" << aborted
           << ",\"queue_peak\":" << queue_peak
           << ",\"slots\":" << config.slot_count
           << ",\"batches\":" << total_batches
           << ",\"overload_policy\":\"" << config.overload_name << "\""
           << ",\"scheduling_policy\":\"" << config.scheduling_name << "\""
           << ",\"host_staging_ms\":" << host_staging_total_ms_
           << ",\"h2d_ms\":" << h2d_total_ms_
           << ",\"preprocess_ms\":" << preprocess_total_ms_
           << ",\"inference_ms\":" << inference_total_ms_
           << ",\"d2h_ms\":" << d2h_total_ms_
           << ",\"cpu_postprocess_ms\":" << postprocess_total_ms_
           << ",\"p50_ms\":" << percentile(latencies_, 0.50)
           << ",\"p90_ms\":" << percentile(latencies_, 0.90)
           << ",\"p99_ms\":" << percentile(latencies_, 0.99)
           << ",\"fps\":" << (elapsed_seconds > 0.0 ? completed_ / elapsed_seconds : 0.0)
           << ",\"environment\":{\"gpu\":\"" << identity.gpu_name
           << "\",\"compute_capability\":\"" << identity.compute_major << '.'
           << identity.compute_minor << "\",\"tensorrt\":\"" << identity.tensorrt_major
           << '.' << identity.tensorrt_minor << '.' << identity.tensorrt_patch
           << "\",\"cuda_runtime\":" << identity.cuda_runtime
           << ",\"cuda_driver\":" << identity.cuda_driver << "},\"batch_distribution\":{";
    bool first = true;
    for (const auto& [size, count] : batch_distribution_) {
        if (!first) output << ',';
        first = false;
        output << '"' << size << "\":" << count;
    }
    output << "},\"batch_sample_storage\":{\"raw_jsonl\":\"batch_timing_samples.jsonl\","
           << "\"retained_in_metrics\":" << kRetainedBatchSamples
           << "},\"per_stream_processed\":{";
    first = true;
    for (const auto& [stream, count] : per_stream_completed_) {
        if (!first) output << ',';
        first = false;
        output << '"' << stream << "\":" << count;
    }
    output << "},\"streams\":[";
    first = true;
    for (const auto& [stream, count] : per_stream_completed_) {
        if (!first) output << ',';
        first = false;
        const auto found = per_stream_latencies_.find(stream);
        const std::vector<double> empty;
        const auto& latencies = found == per_stream_latencies_.end() ? empty : found->second;
        output << "{\"stream_id\":" << stream << ",\"completed\":" << count
               << ",\"fps\":" << (elapsed_seconds > 0.0 ? count / elapsed_seconds : 0.0)
               << ",\"p50_ms\":" << percentile(latencies, 0.50)
               << ",\"p90_ms\":" << percentile(latencies, 0.90)
               << ",\"p99_ms\":" << percentile(latencies, 0.99) << '}';
    }
    output << "],\"batch_samples\":[";
    first = true;
    for (const BatchTimingSample& sample : batches_) {
        if (!first) output << ',';
        first = false;
        output << "{\"batch_id\":" << sample.batch_id
               << ",\"batch_size\":" << sample.batch_size
               << ",\"queue_wait_ms\":" << sample.queue_wait_ms
               << ",\"batch_fill_wait_ms\":" << sample.batch_fill_wait_ms
               << ",\"host_staging_ms\":" << sample.host_staging_ms
               << ",\"capacity_growth_ms\":" << sample.capacity_growth_ms
               << ",\"h2d_ms\":" << sample.h2d_ms
               << ",\"gpu_preprocess_ms\":" << sample.gpu_preprocess_ms
               << ",\"tensorrt_ms\":" << sample.tensorrt_ms
               << ",\"d2h_ms\":" << sample.d2h_ms
               << ",\"cpu_postprocess_ms\":" << sample.cpu_postprocess_ms << '}';
    }
    output << "]}\n";
}

}  // namespace lesson21
