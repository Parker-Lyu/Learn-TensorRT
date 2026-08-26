#pragma once

#include <string>
#include <vector>

namespace lesson31 {

enum class KernelVariant {
    Baseline16x16,
    Block32x8,
    Linear,
    Vectorized,
    Unfused,
};

struct BenchmarkConfig {
    int width{640};
    int height{640};
    int warmup_iterations{50};
    int measured_iterations{500};
};

struct TimingSummary {
    float minimum_ms{0.0F};
    float mean_ms{0.0F};
    float p50_ms{0.0F};
    float p90_ms{0.0F};
    float maximum_ms{0.0F};
};

struct BenchmarkResult {
    KernelVariant variant;
    TimingSummary timing;
    float maximum_absolute_error{0.0F};
    float mean_absolute_error{0.0F};
    int kernel_launches_per_iteration{1};
};

const char* variant_name(KernelVariant variant) noexcept;
KernelVariant parse_variant(const std::string& name);
std::vector<KernelVariant> all_variants();
std::vector<BenchmarkResult> benchmark_variants(
    const BenchmarkConfig& config, const std::vector<KernelVariant>& variants);

}  // namespace lesson31
