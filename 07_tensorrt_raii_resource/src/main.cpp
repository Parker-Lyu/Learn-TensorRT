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

lesson07::InputShape parse_input_shape(std::string_view text) {
    const std::size_t separator = text.find(':');
    if (separator == std::string_view::npos || separator == 0 || separator + 1 >= text.size()) {
        throw std::runtime_error("Input shape must look like NAME:D0xD1x..., got: " +
                                 std::string(text));
    }

    lesson07::InputShape shape;
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
        lesson07::RunConfig config;

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
            } else {
                throw std::runtime_error("Unknown argument: " + std::string(arg));
            }
        }

        const lesson07::InferenceReport report = lesson07::run_smoke_inference(config);

        std::cout << "Engine: " << report.engine_path << '\n';
        std::cout << "Tensor buffers:\n";
        for (const lesson07::TensorReport& tensor : report.tensors) {
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
