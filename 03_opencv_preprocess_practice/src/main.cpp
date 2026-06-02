#include "preprocess.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace {

void print_usage(const char* executable_name) {
    std::cout << "Usage:\n"
              << "  " << executable_name
              << " [image_path] [input_width] [input_height] [output_dir]\n\n"
              << "Defaults:\n"
              << "  image_path: ../assets/dog.webp\n"
              << "  input size: 640 x 640\n"
              << "  output_dir: outputs\n";
}

int parse_positive_int(std::string_view text, const char* name) {
    int value = 0;
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    const auto [parsed_end, error] = std::from_chars(begin, end, value);

    if (text.empty() || error != std::errc() || parsed_end != end) {
        throw std::runtime_error(std::string(name) + " must be a positive integer, got: " +
                                 std::string(text));
    }
    if (value <= 0) {
        throw std::runtime_error(std::string(name) + " must be positive.");
    }
    return value;
}

void write_tensor_preview(const std::filesystem::path& path,
                          const BatchPreprocessResult& result,
                          std::size_t max_values) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to open tensor preview for writing: " + path.string());
    }

    output << "shape: [" << result.batch_size << ", " << result.channels << ", "
           << result.height << ", " << result.width << "]\n";
    output << "layout: NCHW\n";
    output << "dtype: float32\n";
    output << "color: RGB\n";
    output << "range: [0, 1]\n\n";

    const std::size_t value_count = std::min(max_values, result.input_tensor.size());
    for (std::size_t i = 0; i < value_count; ++i) {
        output << i << ": " << result.input_tensor[i] << '\n';
    }

    output.close();
    if (!output) {
        throw std::runtime_error("Failed to write tensor preview: " + path.string());
    }
}

void print_starter_todos() {
    std::cout << "\nTODO 1: implement letterbox_image(...)\n";
    std::cout << "TODO 2: implement append_nchw_rgb_float(...)\n";
    std::cout << "TODO 3: implement preprocess_batch_to_nchw(...)\n";
    std::cout << "TODO 4: implement map_box_to_original_image(...)\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
            print_usage(argv[0]);
            return 0;
        }

        const std::string image_path = argc > 1 ? argv[1] : "../assets/dog.webp";
        const int input_width = argc > 2 ? parse_positive_int(argv[2], "input_width") : 640;
        const int input_height = argc > 3 ? parse_positive_int(argv[3], "input_height") : 640;
        const std::filesystem::path output_dir = argc > 4 ? argv[4] : "outputs";

        const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << image_path << '\n';
            return 1;
        }

        std::filesystem::create_directories(output_dir);

        std::cout << "Image path:     " << image_path << '\n';
        std::cout << "Network input:  " << input_width << " x " << input_height << '\n';
        std::cout << "Output dir:     " << output_dir.string() << '\n';
        std::cout << "Loaded image:   " << image.cols << " x " << image.rows << '\n';

        // TODO 1: Call letterbox_image, save letterbox_debug.jpg, and print scale/padding.
        // TODO 3: Call preprocess_batch_to_nchw and write input_tensor_preview.txt.
        // TODO 4: Call map_box_to_original_image and print one sample mapping.
        (void)write_tensor_preview;

        print_starter_todos();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
