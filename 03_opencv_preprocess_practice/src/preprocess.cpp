#include "preprocess.hpp"

#include <opencv2/imgproc.hpp>

#include <stdexcept>

namespace {

constexpr float kInv255 = 1.0F / 255.0F;

void append_nchw_rgb_float(const cv::Mat& letterboxed_bgr,
                           std::vector<float>& output,
                           int batch_index) {
    if (letterboxed_bgr.empty()) {
        throw std::runtime_error("append_nchw_rgb_float received an empty image.");
    }
    if (letterboxed_bgr.channels() != 3) {
        throw std::runtime_error("append_nchw_rgb_float expects a 3-channel BGR image.");
    }
    if (batch_index < 0) {
        throw std::runtime_error("append_nchw_rgb_float received a negative batch index.");
    }

    (void)output;
    (void)kInv255;

    // TODO 2: Hand-write BGR uint8 HWC -> RGB float32 NCHW conversion here.
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

    // TODO 1: Compute scale, resized shape, padding, resize the image, and copy into gray canvas.
    return cv::Mat();
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

    (void)append_nchw_rgb_float;

    // TODO 3: Allocate result.input_tensor, letterbox each image, and append each tensor slice.
    return result;
}

cv::Rect2f map_box_to_original_image(const cv::Rect2f& box_in_letterbox,
                                     const LetterboxInfo& info) {
    if (info.scale <= 0.0F) {
        throw std::runtime_error("map_box_to_original_image expects a positive letterbox scale.");
    }

    (void)box_in_letterbox;

    // TODO 4: Remove padding, divide by scale, clamp to the original image, and return the box.
    return cv::Rect2f();
}
