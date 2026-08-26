#include "kernel_variants.hpp"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cerr << "CUDA kernel tests skipped: " << cudaGetErrorString(status) << '\n';
        return 77;
    }

    try {
        lesson31::BenchmarkConfig config;
        config.width = 13;
        config.height = 7;
        config.warmup_iterations = 1;
        config.measured_iterations = 3;
        const auto results = lesson31::benchmark_variants(config, lesson31::all_variants());
        if (results.size() != lesson31::all_variants().size()) {
            throw std::runtime_error("not every kernel variant was measured");
        }
        for (const auto& result : results) {
            if (result.maximum_absolute_error > 1e-6F ||
                result.mean_absolute_error > 1e-7F || result.timing.p50_ms <= 0.0F) {
                throw std::runtime_error("kernel variant failed correctness or timing validation");
            }
        }
        bool rejected = false;
        try {
            (void)lesson31::parse_variant("not-a-variant");
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("invalid variant was accepted");
        config.width = std::numeric_limits<int>::max();
        config.height = 2;
        rejected = false;
        try {
            (void)lesson31::benchmark_variants(
                config, {lesson31::KernelVariant::Baseline16x16});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        if (!rejected) throw std::runtime_error("overflowing dimensions were accepted");
        std::cout << "All lesson 31 kernel variants passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
