#include "preprocess.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace lesson11 {
namespace {

constexpr float kInv255 = 1.0F / 255.0F;
const cv::Scalar kLetterboxColor(114, 114, 114);

void write_nchw_rgb_float(const cv::Mat& letterboxed_bgr, std::vector<float>& output) {
    if (letterboxed_bgr.empty() || letterboxed_bgr.channels() != 3) {
        throw std::runtime_error("write_nchw_rgb_float expects a non-empty BGR image.");
    }

    const int height = letterboxed_bgr.rows;
    const int width = letterboxed_bgr.cols;
    const std::size_t plane_stride = static_cast<std::size_t>(height) * width;
    output.resize(3 * plane_stride);

    for (int y = 0; y < height; ++y) {
        const cv::Vec3b* row = letterboxed_bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            const std::size_t hw = static_cast<std::size_t>(y) * width + x;
            output[0 * plane_stride + hw] = row[x][2] * kInv255;
            output[1 * plane_stride + hw] = row[x][1] * kInv255;
            output[2 * plane_stride + hw] = row[x][0] * kInv255;
        }
    }
}

}  // namespace

cv::Mat letterbox_image(const cv::Mat& bgr_image, cv::Size input_size, LetterboxInfo& info) {
    if (bgr_image.empty()) {
        throw std::runtime_error("letterbox_image received an empty image.");
    }
    if (bgr_image.channels() != 3) {
        throw std::runtime_error("letterbox_image expects a 3-channel BGR image.");
    }
    if (input_size.width <= 0 || input_size.height <= 0) {
        throw std::runtime_error("letterbox_image expects a positive target size.");
    }

    info.original_width = bgr_image.cols;
    info.original_height = bgr_image.rows;
    info.input_width = input_size.width;
    info.input_height = input_size.height;

    const float scale_w = static_cast<float>(input_size.width) / bgr_image.cols;
    const float scale_h = static_cast<float>(input_size.height) / bgr_image.rows;
    info.scale = std::min(scale_w, scale_h);
    info.resized_width = std::clamp(static_cast<int>(std::round(bgr_image.cols * info.scale)), 1,
                                    input_size.width);
    info.resized_height = std::clamp(static_cast<int>(std::round(bgr_image.rows * info.scale)), 1,
                                     input_size.height);

    const int pad_width = input_size.width - info.resized_width;
    const int pad_height = input_size.height - info.resized_height;
    info.pad_left = pad_width / 2;
    info.pad_top = pad_height / 2;
    info.pad_right = pad_width - info.pad_left;
    info.pad_bottom = pad_height - info.pad_top;

    cv::Mat resized;
    cv::resize(bgr_image, resized, cv::Size(info.resized_width, info.resized_height), 0, 0,
               cv::INTER_LINEAR);

    cv::Mat letterboxed(input_size, CV_8UC3, kLetterboxColor);
    resized.copyTo(letterboxed(cv::Rect(info.pad_left, info.pad_top, info.resized_width,
                                        info.resized_height)));
    return letterboxed;
}

PreprocessResult preprocess_image(const cv::Mat& bgr_image, cv::Size input_size) {
    PreprocessResult result;
    const cv::Mat letterboxed = letterbox_image(bgr_image, input_size, result.letterbox);
    write_nchw_rgb_float(letterboxed, result.tensor_nchw);
    return result;
}

Box map_box_to_original(const Box& box_in_letterbox, const LetterboxInfo& info) {
    const float x1 = (box_in_letterbox.x1 - static_cast<float>(info.pad_left)) / info.scale;
    const float y1 = (box_in_letterbox.y1 - static_cast<float>(info.pad_top)) / info.scale;
    const float x2 = (box_in_letterbox.x2 - static_cast<float>(info.pad_left)) / info.scale;
    const float y2 = (box_in_letterbox.y2 - static_cast<float>(info.pad_top)) / info.scale;

    Box mapped;
    mapped.x1 = std::clamp(x1, 0.0F, static_cast<float>(info.original_width));
    mapped.y1 = std::clamp(y1, 0.0F, static_cast<float>(info.original_height));
    mapped.x2 = std::clamp(x2, 0.0F, static_cast<float>(info.original_width));
    mapped.y2 = std::clamp(y2, 0.0F, static_cast<float>(info.original_height));
    return mapped;
}

}  // namespace lesson11
