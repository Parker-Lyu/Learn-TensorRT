#include "preprocess.hpp"

#include <opencv2/imgcodecs.hpp>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

namespace {
double mean(const std::vector<double>& values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}
const char* name(lesson20::HostMemoryMode mode) {
    if (mode == lesson20::HostMemoryMode::Pageable) return "pageable";
    if (mode == lesson20::HostMemoryMode::Pinned) return "pinned";
    return "mapped";
}
}

int main(int argc, char** argv) {
    try {
        const auto executable = std::filesystem::absolute(argv[0]).parent_path();
        const auto root = (executable / ".." / "..").lexically_normal();
        std::filesystem::path image_path = root / "assets/img.jpeg";
        int iterations = 30;
        for (int index = 1; index < argc; ++index) {
            const std::string argument = argv[index];
            if (++index >= argc) throw std::invalid_argument(argument + " requires a value");
            if (argument == "--image") image_path = argv[index];
            else if (argument == "--iterations") iterations = std::stoi(argv[index]);
            else throw std::invalid_argument("unknown option: " + argument);
        }
        if (iterations <= 0) throw std::invalid_argument("iterations must be positive");
        const cv::Mat image = cv::imread(image_path.string());
        if (image.empty()) throw std::runtime_error("failed to read image: " + image_path.string());
        const cv::Size target(640, 640);
        std::vector<double> cpu_times;
        std::vector<float> reference;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            reference = lesson20::cpu_resize_bgr_to_rgb_nchw(image, target);
            cpu_times.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
        }
        const auto output_dir = executable / ".." / "outputs";
        std::filesystem::create_directories(output_dir);
        std::ofstream csv(output_dir / "preprocess_benchmark.csv");
        if (!csv) throw std::runtime_error("failed to create benchmark CSV");
        csv << "mode,cpu_ms,host_stage_ms,h2d_ms,gpu_preprocess_ms,d2h_ms,max_abs_error,mean_abs_error\n";

        int device = 0;
        int runtime_version = 0;
        int driver_version = 0;
        cudaDeviceProp properties{};
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
            cudaRuntimeGetVersion(&runtime_version) != cudaSuccess ||
            cudaDriverGetVersion(&driver_version) != cudaSuccess) {
            throw std::runtime_error("failed to query CUDA benchmark environment");
        }
        std::ofstream environment(output_dir / "preprocess_benchmark_environment.json");
        if (!environment) throw std::runtime_error("failed to create benchmark environment JSON");
        environment << "{\n"
                    << "  \"gpu\": \"" << properties.name << "\",\n"
                    << "  \"compute_capability\": \"" << properties.major << '.'
                    << properties.minor << "\",\n"
                    << "  \"cuda_runtime_version\": " << runtime_version << ",\n"
                    << "  \"cuda_driver_version\": " << driver_version << "\n}\n";
        for (const auto mode : {lesson20::HostMemoryMode::Pageable,
                                lesson20::HostMemoryMode::Pinned,
                                lesson20::HostMemoryMode::Mapped}) {
            lesson20::GpuPreprocessor gpu(image.size(), target, mode);
            (void)gpu.run(image);
            std::vector<double> stage, h2d, preprocess, d2h;
            lesson20::GpuPreprocessResult result;
            for (int iteration = 0; iteration < iterations; ++iteration) {
                result = gpu.run(image);
                stage.push_back(result.timing.host_staging_ms);
                h2d.push_back(result.timing.h2d_ms);
                preprocess.push_back(result.timing.gpu_preprocess_ms);
                d2h.push_back(result.timing.d2h_ms);
            }
            double max_error = 0.0, error_sum = 0.0;
            for (std::size_t i = 0; i < reference.size(); ++i) {
                const double error = std::abs(reference[i] - result.tensor_nchw[i]);
                max_error = std::max(max_error, error);
                error_sum += error;
            }
            const double mean_error = error_sum / reference.size();
            std::cout << std::fixed << std::setprecision(4) << name(mode)
                      << " stage=" << mean(stage) << " h2d=" << mean(h2d)
                      << " gpu=" << mean(preprocess) << " d2h=" << mean(d2h)
                      << " max_error=" << max_error << " mean_error=" << mean_error << '\n';
            if (mean_error > 0.02 || max_error > 0.30)
                throw std::runtime_error("GPU preprocessing exceeded numerical tolerance");
            csv << name(mode) << ',' << mean(cpu_times) << ',' << mean(stage) << ',' << mean(h2d)
                << ',' << mean(preprocess) << ',' << mean(d2h) << ',' << max_error << ','
                << mean_error << '\n';
        }
        if (!csv || !environment) throw std::runtime_error("failed to write benchmark evidence");
        std::cout << "GPU=" << properties.name << " compute_capability=" << properties.major << '.'
                  << properties.minor << " CUDA_runtime=" << runtime_version
                  << " CUDA_driver=" << driver_version << '\n';
        std::cout << "cpu=" << mean(cpu_times) << " ms saved "
                  << output_dir / "preprocess_benchmark.csv" << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
