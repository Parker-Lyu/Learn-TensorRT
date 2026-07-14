#include "preprocess.hpp"

#include <opencv2/imgproc.hpp>

#include <chrono>
#include <stdexcept>

namespace lesson17 {

std::vector<float> cpu_resize_bgr_to_rgb_nchw(const cv::Mat& bgr, cv::Size target) {
    if (bgr.empty() || bgr.type() != CV_8UC3)
        throw std::invalid_argument("CPU preprocessing expects a non-empty CV_8UC3 image");
    if (target.width <= 0 || target.height <= 0)
        throw std::invalid_argument("target dimensions must be positive");
    cv::Mat resized;
    cv::resize(bgr, resized, target, 0.0, 0.0, cv::INTER_LINEAR);
    const std::size_t plane = static_cast<std::size_t>(target.width) * target.height;
    std::vector<float> output(3 * plane);
    for (int y = 0; y < target.height; ++y) {
        const auto* row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < target.width; ++x) {
            const std::size_t offset = static_cast<std::size_t>(y) * target.width + x;
            output[offset] = row[x][2] / 255.0F;
            output[plane + offset] = row[x][1] / 255.0F;
            output[2 * plane + offset] = row[x][0] / 255.0F;
        }
    }
    return output;
}

}  // namespace lesson17
