#include "tensorrt_backend.hpp"

#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

lesson21::BatchMetadata metadata(std::uint64_t id, int count, const cv::Mat& image) {
    lesson21::BatchMetadata result;
    result.batch_id = id;
    for (int index = 0; index < count; ++index) {
        result.frames.push_back({0, static_cast<std::uint64_t>(index),
                                 static_cast<std::size_t>(index), lesson21::Clock::now(),
                                 {1.0F, 0.0F, 0.0F, image.cols, image.rows}});
    }
    return result;
}

void print(const lesson21::GpuBatchResult& result) {
    const double checksum = std::accumulate(result.output.begin(), result.output.end(), 0.0);
    std::cout << "batch=" << result.metadata.frames.size()
              << " output_elements=" << result.output.size()
              << " checksum=" << checksum
              << " host_staging_ms=" << result.host_staging_ms
              << " h2d_ms=" << result.h2d_ms
              << " preprocess_ms=" << result.preprocess_ms
              << " inference_ms=" << result.inference_ms
              << " d2h_ms=" << result.d2h_ms << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: integrated_tensorrt_gpu_smoke ENGINE IMAGE [BATCH] [--two-slots]\n";
        return 2;
    }
    try {
        const int batch = argc > 3 ? std::stoi(argv[3]) : 1;
        if (batch < 1 || batch > 4) throw std::invalid_argument("batch must be 1..4");
        cv::Mat image = cv::imread(argv[2]);
        if (image.empty()) throw std::runtime_error("cannot read image");
        lesson21::TensorRtBackend backend(argv[1], 2, {640, 640});
        const std::vector<cv::Mat> images(batch, image);
        const std::size_t first = backend.reserve();
        backend.submit(first, images, metadata(0, batch, image));
        if (argc > 4) {
            const std::size_t second = backend.reserve();
            backend.submit(second, images, metadata(1, batch, image));
            std::cout << "submitted_before_collection=2\n";
            print(backend.collect(second));
        }
        print(backend.collect(first));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
