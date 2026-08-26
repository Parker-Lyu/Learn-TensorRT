#include "kernel_variants.hpp"

#include <cuda_runtime_api.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
    lesson31::BenchmarkConfig benchmark;
    std::vector<lesson31::KernelVariant> variants = lesson31::all_variants();
    std::filesystem::path output{"31_nsight_compute_kernel_analysis/outputs/benchmark.json"};
};

int parse_integer(const char* value, const char* option) {
    try {
        std::size_t consumed = 0;
        const int parsed = std::stoi(value, &consumed);
        if (consumed != std::string(value).size()) throw std::invalid_argument("trailing data");
        return parsed;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " requires an integer");
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        if (index + 1 >= argc) throw std::invalid_argument(option + " requires a value");
        const char* value = argv[++index];
        if (option == "--width") options.benchmark.width = parse_integer(value, "--width");
        else if (option == "--height") options.benchmark.height = parse_integer(value, "--height");
        else if (option == "--warmup")
            options.benchmark.warmup_iterations = parse_integer(value, "--warmup");
        else if (option == "--iterations")
            options.benchmark.measured_iterations = parse_integer(value, "--iterations");
        else if (option == "--variant") {
            options.variants = std::string(value) == "all"
                ? lesson31::all_variants()
                : std::vector<lesson31::KernelVariant>{lesson31::parse_variant(value)};
        } else if (option == "--output") options.output = value;
        else throw std::invalid_argument("unknown option: " + option);
    }
    return options;
}

void write_json(const Options& options, const std::vector<lesson31::BenchmarkResult>& results) {
    if (!options.output.parent_path().empty()) {
        std::filesystem::create_directories(options.output.parent_path());
    }
    std::ofstream output(options.output);
    if (!output) throw std::runtime_error("cannot write benchmark output: " + options.output.string());

    int device = 0;
    int runtime_version = 0;
    int driver_version = 0;
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess ||
        cudaDriverGetVersion(&driver_version) != cudaSuccess) {
        throw std::runtime_error("failed to query CUDA environment");
    }

    output << std::fixed << std::setprecision(8);
    output << "{\n  \"schema_version\": 1,\n"
           << "  \"environment\": {\"gpu\": \"" << properties.name
           << "\", \"compute_capability\": \"" << properties.major << '.' << properties.minor
           << "\", \"cuda_runtime\": " << runtime_version
           << ", \"cuda_driver\": " << driver_version << "},\n"
           << "  \"configuration\": {\"width\": " << options.benchmark.width
           << ", \"height\": " << options.benchmark.height
           << ", \"warmup_iterations\": " << options.benchmark.warmup_iterations
           << ", \"measured_iterations\": " << options.benchmark.measured_iterations << "},\n"
           << "  \"results\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const auto& result = results[index];
        output << "    {\"variant\": \"" << lesson31::variant_name(result.variant)
               << "\", \"kernel_launches_per_iteration\": "
               << result.kernel_launches_per_iteration
               << ", \"timing_ms\": {\"min\": " << result.timing.minimum_ms
               << ", \"mean\": " << result.timing.mean_ms
               << ", \"p50\": " << result.timing.p50_ms
               << ", \"p90\": " << result.timing.p90_ms
               << ", \"max\": " << result.timing.maximum_ms
               << "}, \"maximum_absolute_error\": " << result.maximum_absolute_error
               << ", \"mean_absolute_error\": " << result.mean_absolute_error << '}';
        output << (index + 1 == results.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    if (!output) throw std::runtime_error("failed while writing benchmark output");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto results = lesson31::benchmark_variants(options.benchmark, options.variants);
        write_json(options, results);
        std::cout << std::fixed << std::setprecision(6);
        for (const auto& result : results) {
            std::cout << lesson31::variant_name(result.variant)
                      << " p50_ms=" << result.timing.p50_ms
                      << " p90_ms=" << result.timing.p90_ms
                      << " max_error=" << result.maximum_absolute_error << '\n';
        }
        std::cout << "wrote " << options.output << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
