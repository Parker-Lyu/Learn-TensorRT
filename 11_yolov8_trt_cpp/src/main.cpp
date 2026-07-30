#include "nvtx_range.hpp"
#include "postprocess.hpp"
#include "preprocess.hpp"
#include "tensorrt_runner.hpp"
#include "visualize.hpp"

#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct CliOptions {
    std::string engine_path = "../06_trtexec_engine/outputs/yolov8n_static_fp32.engine";
    std::string image_path = "../assets/img.jpeg";
    std::string output_dir = "outputs";
    float confidence = 0.25F;
    float iou = 0.45F;
    int max_detections = 100;
    int warmup_iterations = 0;
    int iterations = 1;
};

struct LatencySample {
    float preprocess_ms = 0.0F;
    float h2d_ms = 0.0F;
    float enqueue_host_ms = 0.0F;
    float gpu_compute_ms = 0.0F;
    float d2h_ms = 0.0F;
    float postprocess_ms = 0.0F;
    float total_ms = 0.0F;
};

struct PipelineResult {
    std::vector<lesson11::Detection> detections;
    LatencySample latency;
};

void print_usage(const char* executable) {
    std::cout << "Usage:\n"
              << "  " << executable << " [--engine PATH] [--image PATH] [--output-dir DIR]\n"
              << "       [--confidence VALUE] [--iou VALUE] [--max-detections N]\n"
              << "       [--warmup-iterations N] [--iterations N]\n";
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
        } else if (arg == "--warmup-iterations") {
            options.warmup_iterations =
                parse_int(value("--warmup-iterations"), "warmup-iterations");
        } else if (arg == "--iterations") {
            options.iterations = parse_int(value("--iterations"), "iterations");
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (!(options.confidence >= 0.0F && options.confidence <= 1.0F) ||
        !(options.iou >= 0.0F && options.iou <= 1.0F)) {
        throw std::runtime_error("confidence and iou must be in [0, 1].");
    }
    if (options.max_detections <= 0) {
        throw std::runtime_error("max-detections must be positive.");
    }
    if (options.warmup_iterations < 0 || options.iterations <= 0) {
        throw std::runtime_error("warmup-iterations cannot be negative and iterations must be positive.");
    }
    return options;
}

cv::Size input_size_from_shape(const std::vector<int64_t>& shape) {
    if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3) {
        throw std::runtime_error("Expected input shape [1, 3, H, W].");
    }
    return cv::Size(static_cast<int>(shape[3]), static_cast<int>(shape[2]));
}

float milliseconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<float, std::milli>(duration).count();
}

PipelineResult run_pipeline(const cv::Mat& image,
                            lesson11::TensorRtRunner& runner,
                            const lesson11::PostprocessConfig& post_config,
                            const std::string& range_name) {
    lesson11::NvtxRange iteration_range(range_name);
    const auto total_start = std::chrono::steady_clock::now();

    const auto preprocess_start = std::chrono::steady_clock::now();
    lesson11::PreprocessResult preprocessed;
    {
        lesson11::NvtxRange range("preprocess");
        preprocessed = lesson11::preprocess_image(image, input_size_from_shape(runner.input_shape()));
    }
    const auto preprocess_stop = std::chrono::steady_clock::now();

    lesson11::InferenceOutput inference;
    {
        lesson11::NvtxRange range("inference");
        inference = runner.infer(preprocessed.tensor_nchw);
    }

    const auto postprocess_start = std::chrono::steady_clock::now();
    std::vector<lesson11::Detection> detections;
    {
        lesson11::NvtxRange range("postprocess");
        detections = lesson11::decode_yolov8_output(
            inference.values, inference.output_shape, preprocessed.letterbox, post_config);
    }
    const auto postprocess_stop = std::chrono::steady_clock::now();

    PipelineResult result;
    result.detections = std::move(detections);
    result.latency.preprocess_ms = milliseconds(preprocess_stop - preprocess_start);
    result.latency.h2d_ms = inference.h2d_ms;
    result.latency.enqueue_host_ms = inference.enqueue_host_ms;
    result.latency.gpu_compute_ms = inference.gpu_compute_ms;
    result.latency.d2h_ms = inference.d2h_ms;
    result.latency.postprocess_ms = milliseconds(postprocess_stop - postprocess_start);
    result.latency.total_ms = milliseconds(postprocess_stop - total_start);
    return result;
}

void write_json_string(std::ostream& output, const std::string& value) {
    output << '"';
    for (const char ch : value) {
        switch (ch) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                output << ch;
                break;
        }
    }
    output << '"';
}

void write_latency(std::ostream& output, const LatencySample& sample, const std::string& indent) {
    output << indent << "{\n";
    output << indent << "  \"preprocess\": " << sample.preprocess_ms << ",\n";
    output << indent << "  \"h2d\": " << sample.h2d_ms << ",\n";
    output << indent << "  \"enqueue_host\": " << sample.enqueue_host_ms << ",\n";
    output << indent << "  \"gpu_compute\": " << sample.gpu_compute_ms << ",\n";
    output << indent << "  \"d2h\": " << sample.d2h_ms << ",\n";
    output << indent << "  \"postprocess\": " << sample.postprocess_ms << ",\n";
    output << indent << "  \"total\": " << sample.total_ms << '\n';
    output << indent << '}';
}

void write_latency_array(std::ostream& output,
                         const std::vector<LatencySample>& samples,
                         const std::string& indent) {
    output << "[\n";
    for (std::size_t index = 0; index < samples.size(); ++index) {
        write_latency(output, samples[index], indent);
        output << (index + 1 == samples.size() ? "\n" : ",\n");
    }
    output << indent.substr(0, indent.size() - 2) << ']';
}

void write_json(const std::filesystem::path& path,
                const CliOptions& options,
                const std::vector<lesson11::Detection>& detections,
                const std::vector<LatencySample>& warmup_samples,
                const std::vector<LatencySample>& measured_samples) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Failed to write JSON report: " + path.string());
    }
    output << std::fixed << std::setprecision(6);
    output << "{\n  \"engine\": ";
    write_json_string(output, options.engine_path);
    output << ",\n  \"image\": ";
    write_json_string(output, options.image_path);
    output << ",\n  \"warmup_iterations\": " << options.warmup_iterations << ",\n";
    output << "  \"iterations\": " << options.iterations << ",\n";
    output << "  \"latency_ms\": ";
    write_latency(output, measured_samples.back(), "  ");
    output << ",\n  \"warmup_latency_samples_ms\": ";
    write_latency_array(output, warmup_samples, "    ");
    output << ",\n  \"latency_samples_ms\": ";
    write_latency_array(output, measured_samples, "    ");
    output << ",\n  \"detections\": [\n";
    for (std::size_t index = 0; index < detections.size(); ++index) {
        const lesson11::Detection& detection = detections[index];
        output << "    {\"class_id\": " << detection.class_id << ", \"class_name\": ";
        write_json_string(output, detection.class_name);
        output << ", \"confidence\": " << detection.confidence << ", \"box_xyxy\": ["
               << detection.box.x1 << ", " << detection.box.y1 << ", " << detection.box.x2
               << ", " << detection.box.y2 << "]}";
        output << (index + 1 == detections.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

std::string iteration_name(const char* prefix, int index) {
    return std::string(prefix) + '_' + std::to_string(index);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const CliOptions options = parse_args(argc, argv);
        lesson11::PostprocessConfig post_config;
        post_config.confidence_threshold = options.confidence;
        post_config.iou_threshold = options.iou;
        post_config.max_detections = options.max_detections;

        lesson11::TensorRtRunner runner(options.engine_path);
        const cv::Mat image = cv::imread(options.image_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to read image: " + options.image_path);
        }

        std::vector<LatencySample> warmup_samples;
        warmup_samples.reserve(static_cast<std::size_t>(options.warmup_iterations));
        for (int index = 0; index < options.warmup_iterations; ++index) {
            PipelineResult result =
                run_pipeline(image, runner, post_config, iteration_name("warmup_iteration", index));
            warmup_samples.push_back(result.latency);
        }

        std::vector<LatencySample> measured_samples;
        measured_samples.reserve(static_cast<std::size_t>(options.iterations));
        PipelineResult final_result;
        for (int index = 0; index < options.iterations; ++index) {
            final_result =
                run_pipeline(image, runner, post_config, iteration_name("measured_iteration", index));
            measured_samples.push_back(final_result.latency);
        }

        const cv::Mat annotated = lesson11::draw_detections(image, final_result.detections);
        std::filesystem::create_directories(options.output_dir);
        const std::filesystem::path image_path =
            std::filesystem::path(options.output_dir) /
            (std::filesystem::path(options.image_path).stem().string() + "_yolov8_trt_cpp.jpg");
        const std::filesystem::path json_path =
            std::filesystem::path(options.output_dir) / "detections.json";
        if (!cv::imwrite(image_path.string(), annotated)) {
            throw std::runtime_error("Failed to write output image: " + image_path.string());
        }
        write_json(json_path, options, final_result.detections, warmup_samples, measured_samples);

        const LatencySample& last = measured_samples.back();
        std::cout << "Engine: " << options.engine_path << '\n';
        std::cout << "Image: " << options.image_path << '\n';
        std::cout << "Input tensor: " << runner.input_name() << '\n';
        std::cout << "Output tensor: " << runner.output_name() << '\n';
        std::cout << "Warmup iterations: " << options.warmup_iterations << '\n';
        std::cout << "Measured iterations: " << options.iterations << '\n';
        std::cout << "Detections: " << final_result.detections.size() << '\n';
        std::cout << "Last latency ms: preprocess=" << last.preprocess_ms << ", h2d=" << last.h2d_ms
                  << ", enqueue_host=" << last.enqueue_host_ms
                  << ", gpu_compute=" << last.gpu_compute_ms << ", d2h=" << last.d2h_ms
                  << ", postprocess=" << last.postprocess_ms << ", total=" << last.total_ms << '\n';
        std::cout << "Output image: " << image_path.string() << '\n';
        std::cout << "JSON report: " << json_path.string() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
