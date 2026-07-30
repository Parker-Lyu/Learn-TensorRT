#include "tensorrt_raii.hpp"

#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

void print_usage(const char* executable_name) {
    std::cout
        << "Usage:\n"
        << "  " << executable_name << " [--engine PATH] [--input-shape NAME:D0xD1x...]\n"
        << "       [--warmup N] [--iterations N]\n\n"
        << "       [--repeat N] [--inject-failure STAGE] [--memory-tolerance-mib N]\n\n"
        << "Default engine:\n"
        << "  ../06_trtexec_engine/outputs/yolov8n_static_fp32.engine\n\n"
        << "Examples:\n"
        << "  " << executable_name << '\n'
        << "  " << executable_name
        << " --engine ../06_trtexec_engine/outputs/yolov8n_dynamic_fp16.engine \\\n"
        << "      --input-shape images:1x3x640x640\n";
}

int parse_positive_int(std::string_view text, const char* name) {
    int value = 0;
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto [parsed_end, error] = std::from_chars(begin, end, value);

    if (text.empty() || error != std::errc() || parsed_end != end || value <= 0) {
        throw std::runtime_error(std::string(name) + " must be a positive integer, got: " +
                                 std::string(text));
    }
    return value;
}

int32_t parse_dimension(std::string_view text) {
    int32_t value = 0;
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto [parsed_end, error] = std::from_chars(begin, end, value);

    if (text.empty() || error != std::errc() || parsed_end != end || value <= 0) {
        throw std::runtime_error("Input shape dimensions must be positive integers, got: " +
                                 std::string(text));
    }
    return value;
}

lesson08::InputShape parse_input_shape(std::string_view text) {
    const std::size_t separator = text.find(':');
    if (separator == std::string_view::npos || separator == 0 || separator + 1 >= text.size()) {
        throw std::runtime_error("Input shape must look like NAME:D0xD1x..., got: " +
                                 std::string(text));
    }

    lesson08::InputShape shape;
    shape.tensor_name = std::string(text.substr(0, separator));

    std::string_view rest = text.substr(separator + 1);
    while (!rest.empty()) {
        const std::size_t next = rest.find('x');
        const std::string_view token =
            next == std::string_view::npos ? rest : rest.substr(0, next);
        shape.dimensions.push_back(parse_dimension(token));

        if (next == std::string_view::npos) {
            break;
        }
        rest.remove_prefix(next + 1);
    }

    if (shape.dimensions.empty()) {
        throw std::runtime_error("Input shape has no dimensions: " + std::string(text));
    }
    return shape;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        lesson08::RunConfig config;
        int repetitions = 1;
        int memory_tolerance_mib = 16;

        for (int i = 1; i < argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]);
                return 0;
            }

            auto require_value = [&](const char* option) -> std::string_view {
                if (i + 1 >= argc) {
                    throw std::runtime_error(std::string(option) + " requires a value.");
                }
                ++i;
                return std::string_view(argv[i]);
            };

            if (arg == "--engine") {
                config.engine_path = std::string(require_value("--engine"));
            } else if (arg == "--input-shape") {
                config.input_shapes.push_back(parse_input_shape(require_value("--input-shape")));
            } else if (arg == "--warmup") {
                config.warmup_iterations = parse_positive_int(require_value("--warmup"), "warmup");
            } else if (arg == "--iterations") {
                config.measured_iterations =
                    parse_positive_int(require_value("--iterations"), "iterations");
            } else if (arg == "--repeat") {
                repetitions = parse_positive_int(require_value("--repeat"), "repeat");
            } else if (arg == "--inject-failure") {
                config.injected_failure = lesson08::parse_failure_stage(
                    std::string(require_value("--inject-failure")));
            } else if (arg == "--memory-tolerance-mib") {
                memory_tolerance_mib = parse_positive_int(
                    require_value("--memory-tolerance-mib"), "memory-tolerance-mib");
            } else {
                throw std::runtime_error("Unknown argument: " + std::string(arg));
            }
        }

        if (repetitions > 1 || config.injected_failure != lesson08::FailureStage::kNone) {
            lesson08::LifecycleConfig lifecycle_config;
            lifecycle_config.run = config;
            lifecycle_config.repetitions = repetitions;
            lifecycle_config.memory_tolerance_bytes =
                static_cast<std::size_t>(memory_tolerance_mib) * 1024U * 1024U;
            const lesson08::LifecycleReport report =
                lesson08::run_repeated_lifecycle_test(lifecycle_config);
            std::cout << "Lifecycle repetitions: " << report.repetitions << '\n'
                      << "Completed runs: " << report.completed_runs << '\n'
                      << "Expected injected failures: " << report.expected_failures << '\n'
                      << "Device bytes before/after: " << report.device_bytes_before << "/"
                      << report.device_bytes_after << '\n'
                      << "Host RSS bytes before/after: " << report.host_rss_bytes_before << "/"
                      << report.host_rss_bytes_after << '\n'
                      << "Memory stable: " << (report.memory_stable ? "yes" : "no") << '\n';
            return report.memory_stable ? 0 : 1;
        }

        const lesson08::InferenceReport report = lesson08::run_smoke_inference(config);

        std::cout << "Engine: " << report.engine_path << '\n';
        std::cout << "Tensor buffers:\n";
        for (const lesson08::TensorReport& tensor : report.tensors) {
            std::cout << "  - " << tensor.name << " [" << tensor.mode << ", "
                      << tensor.location << ", " << tensor.data_type << "] shape=";
            for (std::size_t dim = 0; dim < tensor.dimensions.size(); ++dim) {
                std::cout << (dim == 0 ? "" : "x") << tensor.dimensions[dim];
            }
            std::cout << " bytes=" << tensor.byte_count << '\n';
        }

        std::cout << "Total device bytes: " << report.total_device_bytes << '\n';
        std::cout << "Average enqueue time: " << report.average_enqueue_ms << " ms\n";
        std::cout << "Smoke inference completed successfully.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
