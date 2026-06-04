#include "cuda_memory_demo.hpp"

#include <charconv>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace {

void print_usage(const char* executable_name) {
    std::cout << "Usage:\n"
              << "  " << executable_name << " [element_count] [iterations]\n\n"
              << "Defaults:\n"
              << "  element_count: 1228800  (1 x 3 x 640 x 640 float32 tensor)\n"
              << "  iterations:    20\n";
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

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
            print_usage(argv[0]);
            return 0;
        }

        DemoConfig config;
        if (argc > 1) {
            config.element_count = parse_positive_size(argv[1], "element_count");
        }
        if (argc > 2) {
            config.iterations = parse_positive_int(argv[2], "iterations");
        }
        if (argc > 3) {
            print_usage(argv[0]);
            return 1;
        }

        return run_cuda_memory_demo(config);
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
