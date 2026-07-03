#include "postprocess.hpp"
#include "preprocess.hpp"
#include "tensorrt_runner.hpp"
#include "visualize.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CliOptions {
    std::string engine_path = "../06_trtexec_engine/outputs/yolov8n_static_fp32.engine";
    std::string image_path = "../assets/dog.webp";
    std::string output_dir = "outputs";
    float confidence = 0.25F;
    float iou = 0.45F;
    int max_detections = 100;
};

void print_usage(const char* executable) {
    std::cout << "Usage:\n"
              << "  " << executable << " [--engine PATH] [--image PATH] [--output-dir DIR]\n"
              << "       [--confidence VALUE] [--iou VALUE] [--max-detections N]\n";
}

float parse_float(const std::string& text, const char* name) {
    std::size_t parsed = 0;
    const float value = std::stof(text, &parsed);
    if (parsed != text.size()) {
        throw std::runtime_error(std::string(name) + " must be a float.");
    }
    return value;
}

int parse_int(const std::string& text, const char* name) {
    std::size_t parsed = 0;
    const int value = std::stoi(text, &parsed);
    if (parsed != text.size()) {
        throw std::runtime_error(std::string(name) + " must be an integer.");
    }
    return value;
}

CliOptions parse_args(int argc, char* argv[]) {
    CliOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        auto value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string(flag) + " requires a value.");
            }
            ++i;
            return argv[i];
        };
        if (arg == "--engine") {
            options.engine_path = value("--engine");
        } else if (arg == "--image") {
            options.image_path = value("--image");
        } else if (arg == "--output-dir") {
            options.output_dir = value("--output-dir");
        } else if (arg == "--confidence") {
            options.confidence = parse_float(value("--confidence"), "confidence");
        } else if (arg == "--iou") {
            options.iou = parse_float(value("--iou"), "iou");
        } else if (arg == "--max-detections") {
            options.max_detections = parse_int(value("--max-detections"), "max-detections");
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    return options;
}

cv::Size input_size_from_shape(const std::vector<int64_t>& shape) {
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3) {
        throw std::runtime_error("Expected input shape [1, 3, H, W].");
    }
    return cv::Size(static_cast<int>(shape[3]), static_cast<int>(shape[2]));
}

void write_json(const std::filesystem::path& path,
                const CliOptions& options,
                const std::vector<lesson10::Detection>& detections,
                float preprocess_ms,
                const lesson10::InferenceOutput& inference,
                float postprocess_ms,
                float total_ms) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to write JSON report: " + path.string());
    }
    output << std::fixed << std::setprecision(6);
    output << "{\n";
    output << "  \"engine\": \"" << options.engine_path << "\",\n";
    output << "  \"image\": \"" << options.image_path << "\",\n";
    output << "  \"latency_ms\": {\n";
    output << "    \"preprocess\": " << preprocess_ms << ",\n";
    output << "    \"h2d\": " << inference.h2d_ms << ",\n";
    output << "    \"enqueue\": " << inference.enqueue_ms << ",\n";
    output << "    \"d2h\": " << inference.d2h_ms << ",\n";
    output << "    \"postprocess\": " << postprocess_ms << ",\n";
    output << "    \"total\": " << total_ms << "\n";
    output << "  },\n";
    output << "  \"detections\": [\n";
    for (std::size_t i = 0; i < detections.size(); ++i) {
        const lesson10::Detection& det = detections[i];
        output << "    {\"class_id\": " << det.class_id << ", \"class_name\": \""
               << det.class_name << "\", \"confidence\": " << det.confidence
               << ", \"box_xyxy\": [" << det.box.x1 << ", " << det.box.y1 << ", "
               << det.box.x2 << ", " << det.box.y2 << "]}";
        output << (i + 1 == detections.size() ? "\n" : ",\n");
    }
    output << "  ]\n";
    output << "}\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const CliOptions options = parse_args(argc, argv);
        lesson10::PostprocessConfig post_config;
        post_config.confidence_threshold = options.confidence;
        post_config.iou_threshold = options.iou;
        post_config.max_detections = options.max_detections;

        lesson10::TensorRtRunner runner(options.engine_path);
        const cv::Mat image = cv::imread(options.image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to read image: " + options.image_path);
        }

        const auto start_total = std::chrono::steady_clock::now();
        const auto start_pre = std::chrono::steady_clock::now();
        const lesson10::PreprocessResult preprocessed =
            lesson10::preprocess_image(image, input_size_from_shape(runner.input_shape()));
        const auto end_pre = std::chrono::steady_clock::now();

        const lesson10::InferenceOutput inference = runner.infer(preprocessed.tensor_nchw);

        const auto start_post = std::chrono::steady_clock::now();
        const std::vector<lesson10::Detection> detections = lesson10::decode_yolov8_output(
            inference.values, inference.output_shape, preprocessed.letterbox, post_config);
        const auto end_post = std::chrono::steady_clock::now();
        const auto end_total = std::chrono::steady_clock::now();

        const cv::Mat annotated = lesson10::draw_detections(image, detections);
        std::filesystem::create_directories(options.output_dir);
        const std::filesystem::path image_path =
            std::filesystem::path(options.output_dir) / "dog_yolov8_trt_cpp.jpg";
        const std::filesystem::path json_path =
            std::filesystem::path(options.output_dir) / "detections.json";
        if (!cv::imwrite(image_path.string(), annotated)) {
            throw std::runtime_error("Failed to write output image: " + image_path.string());
        }

        const auto ms = [](auto duration) {
            return std::chrono::duration<float, std::milli>(duration).count();
        };
        const float preprocess_ms = ms(end_pre - start_pre);
        const float postprocess_ms = ms(end_post - start_post);
        const float total_ms = ms(end_total - start_total);
        write_json(json_path, options, detections, preprocess_ms, inference, postprocess_ms,
                   total_ms);

        std::cout << "Engine: " << options.engine_path << '\n';
        std::cout << "Image: " << options.image_path << '\n';
        std::cout << "Input tensor: " << runner.input_name() << '\n';
        std::cout << "Output tensor: " << runner.output_name() << '\n';
        std::cout << "Detections: " << detections.size() << '\n';
        for (const lesson10::Detection& det : detections) {
            std::cout << "  " << det.class_name << " " << det.confidence << " box=["
                      << det.box.x1 << ", " << det.box.y1 << ", " << det.box.x2 << ", "
                      << det.box.y2 << "]\n";
        }
        std::cout << "Latency ms: preprocess=" << preprocess_ms << ", h2d=" << inference.h2d_ms
                  << ", enqueue=" << inference.enqueue_ms << ", d2h=" << inference.d2h_ms
                  << ", postprocess=" << postprocess_ms << ", total=" << total_ms << '\n';
        std::cout << "Output image: " << image_path.string() << '\n';
        std::cout << "JSON report: " << json_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
