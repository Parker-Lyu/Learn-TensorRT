#include "preprocess.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

void print_usage(const char* executable_name) {
    std::cout << "Usage:\n"
              << "  " << executable_name
              << " [image_path] [input_width] [input_height] [output_dir]\n\n"
              << "Defaults:\n"
              << "  image_path: ../assets/img.jpeg\n"
              << "  input size: 640 x 640\n"
              << "  output_dir: outputs\n";
}

int parse_positive_int(std::string_view text, const char* name) {
    int value = 0;
    const char* begin = text.data();
    const char* end = text.data() + text.size();
    // from_chars avoids locale rules and exceptions, which keeps CLI parsing predictable.
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

void write_tensor_binary(const std::filesystem::path& path, const std::vector<float>& tensor) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("Failed to open tensor binary for writing: " + path.string());
    }

    // Write raw float32 bytes exactly as a TensorRT input buffer would be laid out in memory.
    output.write(reinterpret_cast<const char*>(tensor.data()),
                 static_cast<std::streamsize>(tensor.size() * sizeof(float)));
    output.close();
    if (!output) {
        throw std::runtime_error("Failed to write tensor binary: " + path.string());
    }
}

void write_tensor_preview(const std::filesystem::path& path,
                          const BatchPreprocessResult& result,
                          size_t max_values) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to open tensor preview for writing: " + path.string());
    }

    // The preview is intentionally text so learners can inspect layout without a debugger.
    output << "shape: [" << result.batch_size << ", " << result.channels << ", " << result.height
           << ", " << result.width << "]\n";
    output << "layout: NCHW\n";
    output << "dtype: float32\n";
    output << "color: RGB\n";
    output << "range: [0, 1]\n\n";

    const size_t value_count = std::min(max_values, result.input_tensor.size());
    output << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < value_count; ++i) {
        output << i << ": " << result.input_tensor[i] << '\n';
    }

    output.close();
    if (!output) {
        throw std::runtime_error("Failed to write tensor preview: " + path.string());
    }
}

void print_letterbox_info(const LetterboxInfo& info) {
    std::cout << "Original image: " << info.original_width << " x " << info.original_height
              << '\n';
    std::cout << "Network input:  " << info.input_width << " x " << info.input_height << '\n';
    std::cout << "Resized image:  " << info.resized_width << " x " << info.resized_height
              << '\n';
    std::cout << "Scale:          " << info.scale << '\n';
    std::cout << "Padding:        left=" << info.pad_left << ", top=" << info.pad_top
              << ", right=" << info.pad_right << ", bottom=" << info.pad_bottom << '\n';
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        if (argc > 5) {
            print_usage(argv[0]);
            throw std::invalid_argument("Too many command-line arguments.");
        }
        if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
            print_usage(argv[0]);
            return 0;
        }

        const std::string image_path = argc > 1 ? argv[1] : "../assets/img.jpeg";
        const int input_width = argc > 2 ? parse_positive_int(argv[2], "input_width") : 640;
        const int input_height = argc > 3 ? parse_positive_int(argv[3], "input_height") : 640;
        const std::filesystem::path output_dir = argc > 4 ? argv[4] : "outputs";

        const cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to read image: " << image_path << '\n';
            return 1;
        }

        std::filesystem::create_directories(output_dir);

        // Save the letterboxed image separately so resize and padding can be checked visually.
        LetterboxInfo debug_info;
        const cv::Mat letterboxed = letterbox_image(image, cv::Size(input_width, input_height),
                                                    debug_info);
        const std::filesystem::path debug_image_path = output_dir / "letterbox_debug.jpg";
        if (!cv::imwrite(debug_image_path.string(), letterboxed)) {
            throw std::runtime_error("Failed to write debug image: " + debug_image_path.string());
        }

        // Run the complete preprocessing path used for inference inputs.
        const BatchPreprocessResult result =
            preprocess_batch_to_nchw(std::vector<cv::Mat>{image}, cv::Size(input_width, input_height));
        const std::filesystem::path tensor_bin_path = output_dir / "input_tensor_nchw_float32.bin";
        const std::filesystem::path tensor_preview_path = output_dir / "input_tensor_preview.txt";
        write_tensor_binary(tensor_bin_path, result.input_tensor);
        write_tensor_preview(tensor_preview_path, result, 96);

        const LetterboxInfo& info = result.letterbox_infos.front();
        print_letterbox_info(info);

        std::cout << "Tensor shape:   [" << result.batch_size << ", " << result.channels << ", "
                  << result.height << ", " << result.width << "]\n";
        std::cout << "Tensor values:  " << result.input_tensor.size() << " float32 values, "
                  << result.input_tensor.size() * sizeof(float) << " bytes\n";
        std::cout << "Debug image:    " << debug_image_path.string() << '\n';
        std::cout << "Tensor binary:  " << tensor_bin_path.string() << '\n';
        std::cout << "Tensor preview: " << tensor_preview_path.string() << '\n';

        // A proportional sample box remains inside any valid input shape and demonstrates how
        // model outputs are recovered after letterbox.
        const cv::Rect2f sample_box(0.25F * static_cast<float>(input_width),
                                    0.25F * static_cast<float>(input_height),
                                    0.50F * static_cast<float>(input_width),
                                    0.50F * static_cast<float>(input_height));
        const cv::Rect2f original_box = map_box_to_original_image(sample_box, info);
        std::cout << "Sample box in network input: x=" << sample_box.x << ", y=" << sample_box.y
                  << ", w=" << sample_box.width << ", h=" << sample_box.height << '\n';
        std::cout << "Mapped back to original:     x=" << original_box.x << ", y="
                  << original_box.y << ", w=" << original_box.width
                  << ", h=" << original_box.height << '\n';

        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
