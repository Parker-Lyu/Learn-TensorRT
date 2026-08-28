#include "mlp_layernorm.hpp"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA MLP tests skipped: " << cudaGetErrorString(status) << '\n';
        return 77;
    }

    try {
        lesson31::BenchmarkConfig config;
        config.rows = 7;
        config.input_features = 5;
        config.hidden_features = 37;
        config.output_features = 3;
        config.warmup_iterations = 1;
        config.measured_iterations = 2;
        const auto results = lesson31::benchmark_network(config, lesson31::all_variants());
        if (results.size() != 2) throw std::runtime_error("both variants must run");
        for (const auto& result : results) {
            if (result.maximum_absolute_error > 2.0e-4F ||
                result.mean_absolute_error > 2.0e-5F ||
                result.layernorm_timing.p50_ms <= 0.0F ||
                result.network_timing.p50_ms <= 0.0F ||
                result.reduction_block_size <= 0) {
                throw std::runtime_error("variant failed correctness, timing, or launch validation");
            }
        }
        if (results[0].layernorm_launches != 2 || results[1].layernorm_launches != 1) {
            throw std::runtime_error("unexpected LayerNorm launch count");
        }
        bool rejected = false;
        try {
            (void)lesson31::parse_variant("unknown");
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("unknown variant was accepted");
        config.rows = 0;
        rejected = false;
        try {
            (void)lesson31::benchmark_network(config, lesson31::all_variants());
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("invalid dimensions were accepted");
        std::cout << "All lesson 31 MLP and LayerNorm checks passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
