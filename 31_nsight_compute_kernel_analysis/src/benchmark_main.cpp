#include "mlp_layernorm.hpp"

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
    std::vector<lesson31::LayerNormVariant> variants = lesson31::all_variants();
    std::filesystem::path output{"31_nsight_compute_kernel_analysis/outputs/mlp_benchmark.json"};
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
        if (option == "--rows") options.benchmark.rows = parse_integer(value, "--rows");
        else if (option == "--input-features")
            options.benchmark.input_features = parse_integer(value, "--input-features");
        else if (option == "--hidden-features")
            options.benchmark.hidden_features = parse_integer(value, "--hidden-features");
        else if (option == "--output-features")
            options.benchmark.output_features = parse_integer(value, "--output-features");
        else if (option == "--warmup")
            options.benchmark.warmup_iterations = parse_integer(value, "--warmup");
        else if (option == "--iterations")
            options.benchmark.measured_iterations = parse_integer(value, "--iterations");
        else if (option == "--scope") {
            const std::string scope = value;
            if (scope == "both") options.benchmark.scope = lesson31::MeasurementScope::Both;
            else if (scope == "layernorm")
                options.benchmark.scope = lesson31::MeasurementScope::LayerNorm;
            else if (scope == "network")
                options.benchmark.scope = lesson31::MeasurementScope::Network;
            else throw std::invalid_argument("--scope requires both, layernorm, or network");
        }
        else if (option == "--variant") {
            options.variants = std::string(value) == "all"
                ? lesson31::all_variants()
                : std::vector<lesson31::LayerNormVariant>{lesson31::parse_variant(value)};
        } else if (option == "--output") {
            options.output = value;
        } else {
            throw std::invalid_argument("unknown option: " + option);
        }
    }
    return options;
}

void write_timing(std::ostream& output, const lesson31::TimingSummary& timing) {
    output << "{\"min\": " << timing.minimum_ms
           << ", \"mean\": " << timing.mean_ms
           << ", \"p50\": " << timing.p50_ms
           << ", \"p90\": " << timing.p90_ms
           << ", \"max\": " << timing.maximum_ms << '}';
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
    output << "{\n  \"schema_version\": 2,\n"
           << "  \"environment\": {\"gpu\": \"" << properties.name
           << "\", \"compute_capability\": \"" << properties.major << '.' << properties.minor
           << "\", \"cuda_runtime\": " << runtime_version
           << ", \"cuda_driver\": " << driver_version << "},\n"
           << "  \"configuration\": {\"rows\": " << options.benchmark.rows
           << ", \"input_features\": " << options.benchmark.input_features
           << ", \"hidden_features\": " << options.benchmark.hidden_features
           << ", \"output_features\": " << options.benchmark.output_features
           << ", \"epsilon\": " << options.benchmark.epsilon
           << ", \"warmup_iterations\": " << options.benchmark.warmup_iterations
           << ", \"measured_iterations\": " << options.benchmark.measured_iterations << "},\n"
           << "  \"results\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const auto& result = results[index];
        output << "    {\"variant\": \"" << lesson31::variant_name(result.variant)
               << "\", \"layernorm_launches\": " << result.layernorm_launches
               << ", \"launch_configuration\": {\"reduction_block_size\": "
               << result.reduction_block_size << ", \"apply_block_size\": "
               << result.apply_block_size << "}, \"layernorm_timing_ms\": ";
        write_timing(output, result.layernorm_timing);
        output << ", \"network_timing_ms\": ";
        write_timing(output, result.network_timing);
        output << ", \"maximum_absolute_error\": " << result.maximum_absolute_error
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
        const auto results = lesson31::benchmark_network(options.benchmark, options.variants);
        write_json(options, results);
        std::cout << std::fixed << std::setprecision(6);
        for (const auto& result : results) {
            std::cout << lesson31::variant_name(result.variant)
                      << " layernorm_p50_ms=" << result.layernorm_timing.p50_ms
                      << " network_p50_ms=" << result.network_timing.p50_ms
                      << " max_error=" << result.maximum_absolute_error
                      << " reduction_block=" << result.reduction_block_size << '\n';
        }
        std::cout << "wrote " << options.output << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
