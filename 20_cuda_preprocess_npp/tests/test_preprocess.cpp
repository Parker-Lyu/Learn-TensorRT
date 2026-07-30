#include "preprocess.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }

int main() {
    try {
        cv::Mat image(2, 2, CV_8UC3);
        image.at<cv::Vec3b>(0, 0) = {0, 0, 255};
        image.at<cv::Vec3b>(0, 1) = {0, 255, 0};
        image.at<cv::Vec3b>(1, 0) = {255, 0, 0};
        image.at<cv::Vec3b>(1, 1) = {255, 255, 255};
        const auto cpu = lesson20::cpu_resize_bgr_to_rgb_nchw(image, image.size());
        require(cpu.size() == 12, "CPU NCHW size mismatch");
        require(cpu[0] == 1.0F && cpu[4] == 0.0F && cpu[8] == 0.0F,
                "CPU BGR to RGB layout mismatch");
        for (const auto mode : {lesson20::HostMemoryMode::Pageable,
                                lesson20::HostMemoryMode::Pinned,
                                lesson20::HostMemoryMode::Mapped}) {
            lesson20::GpuPreprocessor gpu(image.size(), image.size(), mode);
            const auto result = gpu.run(image);
            for (std::size_t index = 0; index < cpu.size(); ++index)
                require(std::abs(cpu[index] - result.tensor_nchw[index]) < 1e-6F,
                        "GPU result differs from exact CPU reference");
        }
        bool threw = false;
        try { (void)lesson20::cpu_resize_bgr_to_rgb_nchw({}, {2, 2}); } catch (...) { threw = true; }
        require(threw, "empty input was accepted");
        std::cout << "All CUDA preprocessing tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
