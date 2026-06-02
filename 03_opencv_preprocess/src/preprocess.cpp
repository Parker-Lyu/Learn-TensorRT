#include "preprocess.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

constexpr float kInv255 = 1.0F / 255.0F;
const cv::Scalar kLetterboxColor(114, 114, 114);

void append_nchw_rgb_float(const cv::Mat& letterboxed_bgr,
                           std::vector<float>& output,
                           int batch_index) {
    if (letterboxed_bgr.empty() || letterboxed_bgr.channels() != 3) {
        throw std::runtime_error("append_nchw_rgb_float expects a non-empty 3-channel image.");
    }

    const int height = letterboxed_bgr.rows;
    const int width = letterboxed_bgr.cols;
    const std::size_t plane_stride = static_cast<std::size_t>(height) * width;
    const std::size_t image_stride = 3 * plane_stride;
    const std::size_t batch_offset = static_cast<std::size_t>(batch_index) * image_stride;
    if (batch_offset + image_stride > output.size()) {
        throw std::runtime_error("Output tensor is too small for the requested batch index.");
    }

    for (int y = 0; y < height; ++y) {
        const cv::Vec3b* row = letterboxed_bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            const cv::Vec3b& bgr = row[x];
            const std::size_t hw_index = static_cast<std::size_t>(y) * width + x;

            output[batch_offset + 0 * plane_stride + hw_index] = bgr[2] * kInv255;
            output[batch_offset + 1 * plane_stride + hw_index] = bgr[1] * kInv255;
            output[batch_offset + 2 * plane_stride + hw_index] = bgr[0] * kInv255;
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

    const float scale_w = static_cast<float>(input_size.width) / static_cast<float>(bgr_image.cols);
    const float scale_h = static_cast<float>(input_size.height) / static_cast<float>(bgr_image.rows);
    info.scale = std::min(scale_w, scale_h);

    info.resized_width = static_cast<int>(std::round(bgr_image.cols * info.scale));
    info.resized_height = static_cast<int>(std::round(bgr_image.rows * info.scale));
    info.resized_width = std::max(1, std::min(info.resized_width, input_size.width));
    info.resized_height = std::max(1, std::min(info.resized_height, input_size.height));

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
    const cv::Rect roi(info.pad_left, info.pad_top, info.resized_width, info.resized_height);
    resized.copyTo(letterboxed(roi));

    return letterboxed;
}

BatchPreprocessResult preprocess_batch_to_nchw(const std::vector<cv::Mat>& bgr_images,
                                               cv::Size input_size) {
    if (bgr_images.empty()) {
        throw std::runtime_error("preprocess_batch_to_nchw requires at least one image.");
    }
    if (input_size.width <= 0 || input_size.height <= 0) {
        throw std::runtime_error("preprocess_batch_to_nchw expects a positive target size.");
    }

    BatchPreprocessResult result;
    result.batch_size = static_cast<int>(bgr_images.size());
    result.height = input_size.height;
    result.width = input_size.width;
    result.letterbox_infos.reserve(bgr_images.size());
    result.input_tensor.resize(static_cast<size_t>(result.batch_size) * result.channels *
                               result.height * result.width);

    for (int batch_index = 0; batch_index < result.batch_size; ++batch_index) {
        LetterboxInfo info;
        const cv::Mat letterboxed = letterbox_image(bgr_images[batch_index], input_size, info);
        result.letterbox_infos.push_back(info);
        append_nchw_rgb_float(letterboxed, result.input_tensor, batch_index);
    }

    return result;
}

cv::Rect2f map_box_to_original_image(const cv::Rect2f& box_in_letterbox,
                                     const LetterboxInfo& info) {
    const float x1 = (box_in_letterbox.x - static_cast<float>(info.pad_left)) / info.scale;
    const float y1 = (box_in_letterbox.y - static_cast<float>(info.pad_top)) / info.scale;
    const float x2 = (box_in_letterbox.x + box_in_letterbox.width -
                      static_cast<float>(info.pad_left)) /
                     info.scale;
    const float y2 = (box_in_letterbox.y + box_in_letterbox.height -
                      static_cast<float>(info.pad_top)) /
                     info.scale;

    const float clamped_x1 = std::clamp(x1, 0.0F, static_cast<float>(info.original_width));
    const float clamped_y1 = std::clamp(y1, 0.0F, static_cast<float>(info.original_height));
    const float clamped_x2 = std::clamp(x2, 0.0F, static_cast<float>(info.original_width));
    const float clamped_y2 = std::clamp(y2, 0.0F, static_cast<float>(info.original_height));

    return cv::Rect2f(clamped_x1, clamped_y1, clamped_x2 - clamped_x1, clamped_y2 - clamped_y1);
}
