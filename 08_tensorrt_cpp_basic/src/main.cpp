#include "tensorrt_basic.hpp"

#include <charconv>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace {

void print_usage(const char* executable_name) {
    std::cout
        << "Usage:\n"
        << "  " << executable_name << " [--onnx PATH] [--engine PATH] [--load-engine]\n"
        << "       [--timing-cache PATH] [--fp16] [--workspace-mib N]\n"
        << "       [--input-shape NAME:D0xD1x...]\n"
        << "       [--warmup N] [--iterations N]\n\n"
        << "Defaults:\n"
        << "  --onnx          ../05_torch_to_onnx/outputs/yolov8n.onnx\n"
        << "  --engine        outputs/yolov8n_cpp_basic.engine\n"
        << "  --timing-cache  outputs/tensorrt_timing_fp32.cache, or\n"
        << "                  outputs/tensorrt_timing_fp16.cache with --fp16\n\n"
        << "Examples:\n"
        << "  " << executable_name << '\n'
        << "  " << executable_name << " --fp16 --engine outputs/yolov8n_cpp_basic_fp16.engine\n"
        << "  " << executable_name << " --load-engine --engine outputs/yolov8n_cpp_basic.engine\n"
        << "  " << executable_name
        << " --onnx ../05_torch_to_onnx/outputs/yolov8n_dynamic.onnx \\\n"
        << "      --engine outputs/yolov8n_dynamic_cpp_basic.engine \\\n"
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

std::size_t parse_positive_size(std::string_view text, const char* name) {
    std::size_t value = 0;
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto [parsed_end, error] = std::from_chars(begin, end, value);
    if (text.empty() || error != std::errc() || parsed_end != end || value == 0) {
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

    return shape;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        lesson08::AppConfig config;

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

            if (arg == "--onnx") {
                config.onnx_path = std::string(require_value("--onnx"));
            } else if (arg == "--engine") {
                config.engine_path = std::string(require_value("--engine"));
            } else if (arg == "--timing-cache") {
                config.timing_cache_path = std::string(require_value("--timing-cache"));
            } else if (arg == "--load-engine") {
                config.load_engine_only = true;
            } else if (arg == "--fp16") {
                config.enable_fp16 = true;
            } else if (arg == "--workspace-mib") {
                config.workspace_mib = parse_positive_size(require_value("--workspace-mib"),
                                                           "workspace-mib");
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

        const lesson08::AppReport report = lesson08::run_tensorrt_cpp_basic(config);

        std::cout << "ONNX: " << (report.onnx_path.empty() ? "(not used)" : report.onnx_path)
                  << '\n';
        std::cout << "Engine: " << report.engine_path << '\n';
        std::cout << "Engine source: " << (report.engine_built ? "built from ONNX" : "loaded")
                  << '\n';
        std::cout << "Engine bytes: " << report.engine_bytes << '\n';
        if (!report.timing_cache_path.empty()) {
            std::cout << "Timing cache: " << report.timing_cache_path << " ("
                      << (report.timing_cache_loaded ? "loaded" : "created") << ", "
                      << (report.timing_cache_written ? "written" : "not written")
                      << ", bytes=" << report.timing_cache_bytes << ")\n";
        }
        std::cout << "FP16 requested and enabled: " << (report.fp16_enabled ? "yes" : "no")
                  << '\n';
        std::cout << "Tensor buffers:\n";
        for (const lesson08::TensorSummary& tensor : report.tensors) {
            std::cout << "  - " << tensor.name << " [" << tensor.mode << ", "
                      << tensor.location << ", " << tensor.data_type << "] shape=";
            for (std::size_t i = 0; i < tensor.dimensions.size(); ++i) {
                std::cout << (i == 0 ? "" : "x") << tensor.dimensions[i];
            }
            std::cout << " bytes=" << tensor.byte_count;
            if (tensor.mode == "output") {
                std::cout << " checksum=" << tensor.output_checksum;
            }
            std::cout << '\n';
        }
        std::cout << "Total device bytes: " << report.total_device_bytes << '\n';
        std::cout << "Total pinned host bytes: " << report.total_host_bytes << '\n';
        std::cout << "Average enqueue time: " << report.average_enqueue_ms << " ms\n";
        std::cout << "C++ TensorRT basic flow completed successfully.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
