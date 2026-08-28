#pragma once

#include <string>
#include <vector>

namespace lesson31 {

enum class LayerNormVariant {
    Baseline,
    Fused,
};

enum class MeasurementScope {
    Both,
    LayerNorm,
    Network,
};

struct BenchmarkConfig {
    int rows{2048};
    int input_features{64};
    int hidden_features{128};
    int output_features{32};
    int warmup_iterations{20};
    int measured_iterations{100};
    float epsilon{1.0e-5F};
    MeasurementScope scope{MeasurementScope::Both};
};

struct TimingSummary {
    float minimum_ms{0.0F};
    float mean_ms{0.0F};
    float p50_ms{0.0F};
    float p90_ms{0.0F};
    float maximum_ms{0.0F};
};

struct BenchmarkResult {
    LayerNormVariant variant;
    TimingSummary layernorm_timing;
    TimingSummary network_timing;
    float maximum_absolute_error{0.0F};
    float mean_absolute_error{0.0F};
    int layernorm_launches{0};
    int reduction_block_size{0};
    int apply_block_size{0};
};

const char* variant_name(LayerNormVariant variant) noexcept;
LayerNormVariant parse_variant(const std::string& name);
std::vector<LayerNormVariant> all_variants();
std::vector<BenchmarkResult> benchmark_network(
    const BenchmarkConfig& config, const std::vector<LayerNormVariant>& variants);

}  // namespace lesson31
