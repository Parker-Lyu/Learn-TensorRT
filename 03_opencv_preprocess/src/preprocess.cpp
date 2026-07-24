#include "preprocess.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

constexpr float kInv255 = 1.0F / 255.0F;
const cv::Scalar kLetterboxColor(114, 114, 114);

void validate_bgr_u8_image(const cv::Mat& image, const char* caller) {
    if (image.empty()) {
        throw std::invalid_argument(std::string(caller) + " received an empty image.");
    }
    if (image.type() != CV_8UC3) {
        throw std::invalid_argument(std::string(caller) +
                                    " expects a CV_8UC3 BGR image.");
    }
}

std::size_t checked_tensor_element_count(std::size_t batch_size, cv::Size input_size) {
    constexpr std::size_t channels = 3;
    std::size_t count = batch_size;
    for (const std::size_t factor : {channels, static_cast<std::size_t>(input_size.height),
                                     static_cast<std::size_t>(input_size.width)}) {
        if (factor != 0 && count > std::numeric_limits<std::size_t>::max() / factor) {
            throw std::overflow_error("Requested input tensor is too large.");
        }
        count *= factor;
    }
    return count;
}

void validate_letterbox_info(const LetterboxInfo& info) {
    if (info.original_width <= 0 || info.original_height <= 0 || info.input_width <= 0 ||
        info.input_height <= 0 || info.resized_width <= 0 || info.resized_height <= 0 ||
        !std::isfinite(info.scale) || info.scale <= 0.0F || info.pad_left < 0 ||
        info.pad_top < 0 || info.pad_right < 0 || info.pad_bottom < 0 ||
        info.resized_width + info.pad_left + info.pad_right != info.input_width ||
        info.resized_height + info.pad_top + info.pad_bottom != info.input_height) {
        throw std::invalid_argument("LetterboxInfo is invalid or internally inconsistent.");
    }
}

// Convert one OpenCV BGR HWC image into the batch tensor as RGB float32 NCHW.
void append_nchw_rgb_float(const cv::Mat& letterboxed_bgr,
                           std::vector<float>& output,
                           int batch_index) {
    validate_bgr_u8_image(letterboxed_bgr, "append_nchw_rgb_float");

    const int height = letterboxed_bgr.rows;
    const int width = letterboxed_bgr.cols;
    // One channel plane stores all H * W pixels for a single color channel.
    const std::size_t plane_stride = static_cast<std::size_t>(height) * width;
    const std::size_t image_stride = 3 * plane_stride;
    const std::size_t batch_offset = static_cast<std::size_t>(batch_index) * image_stride;
    if (batch_offset + image_stride > output.size()) {
        throw std::out_of_range("Output tensor is too small for the requested batch index.");
    }

    for (int y = 0; y < height; ++y) {
        const cv::Vec3b* row = letterboxed_bgr.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            const cv::Vec3b& bgr = row[x];
            const std::size_t hw_index = static_cast<std::size_t>(y) * width + x;

            // OpenCV reads BGR uint8. TensorRT examples commonly feed RGB float values in [0, 1].
            output[batch_offset + 0 * plane_stride + hw_index] = bgr[2] * kInv255;
            output[batch_offset + 1 * plane_stride + hw_index] = bgr[1] * kInv255;
            output[batch_offset + 2 * plane_stride + hw_index] = bgr[0] * kInv255;
        }
    }
}

}  // namespace

cv::Mat letterbox_image(const cv::Mat& bgr_image, cv::Size input_size, LetterboxInfo& info) {
    validate_bgr_u8_image(bgr_image, "letterbox_image");
    if (input_size.width <= 0 || input_size.height <= 0) {
        throw std::invalid_argument("letterbox_image expects a positive target size.");
    }

    info.original_width = bgr_image.cols;
    info.original_height = bgr_image.rows;
    info.input_width = input_size.width;
    info.input_height = input_size.height;

    const float scale_w = static_cast<float>(input_size.width) / static_cast<float>(bgr_image.cols);
    const float scale_h = static_cast<float>(input_size.height) / static_cast<float>(bgr_image.rows);
    // Use the smaller scale so the resized image fits inside the input without cropping.
    info.scale = std::min(scale_w, scale_h);

    info.resized_width = static_cast<int>(std::round(bgr_image.cols * info.scale));
    info.resized_height = static_cast<int>(std::round(bgr_image.rows * info.scale));
    // Rounding can produce edge cases on tiny images; clamp to a valid ROI inside the input.
    info.resized_width = std::max(1, std::min(info.resized_width, input_size.width));
    info.resized_height = std::max(1, std::min(info.resized_height, input_size.height));

    const int pad_width = input_size.width - info.resized_width;
    const int pad_height = input_size.height - info.resized_height;
    info.pad_left = pad_width / 2;
    info.pad_top = pad_height / 2;
    // Put any leftover pixel on the right/bottom so all padding still sums to the target size.
    info.pad_right = pad_width - info.pad_left;
    info.pad_bottom = pad_height - info.pad_top;

    cv::Mat resized;
    cv::resize(bgr_image, resized, cv::Size(info.resized_width, info.resized_height), 0, 0,
               cv::INTER_LINEAR);

    cv::Mat letterboxed(input_size, CV_8UC3, kLetterboxColor);
    const cv::Rect roi(info.pad_left, info.pad_top, info.resized_width, info.resized_height);
    // The ROI is where the resized image lives inside the padded network input.
    resized.copyTo(letterboxed(roi));

    return letterboxed;
}

BatchPreprocessResult preprocess_batch_to_nchw(const std::vector<cv::Mat>& bgr_images,
                                               cv::Size input_size) {
    if (bgr_images.empty()) {
        throw std::invalid_argument("preprocess_batch_to_nchw requires at least one image.");
    }
    if (input_size.width <= 0 || input_size.height <= 0) {
        throw std::invalid_argument("preprocess_batch_to_nchw expects a positive target size.");
    }
    if (bgr_images.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("Batch size does not fit the lesson's int shape metadata.");
    }

    BatchPreprocessResult result;
    result.batch_size = static_cast<int>(bgr_images.size());
    result.height = input_size.height;
    result.width = input_size.width;
    result.letterbox_infos.reserve(bgr_images.size());
    // Allocate the full batch tensor once so each image can write directly to its NCHW slot.
    result.input_tensor.resize(checked_tensor_element_count(bgr_images.size(), input_size));

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
    validate_letterbox_info(info);
    if (!std::isfinite(box_in_letterbox.x) || !std::isfinite(box_in_letterbox.y) ||
        !std::isfinite(box_in_letterbox.width) || !std::isfinite(box_in_letterbox.height) ||
        box_in_letterbox.width < 0.0F || box_in_letterbox.height < 0.0F) {
        throw std::invalid_argument("The letterbox-space box must be finite and non-negative.");
    }

    // Undo letterbox in reverse order: remove padding first, then divide by the resize scale.
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

    // Clamping may shrink boxes that touch padding or extend beyond the image boundary.
    return cv::Rect2f(clamped_x1, clamped_y1, std::max(0.0F, clamped_x2 - clamped_x1),
                      std::max(0.0F, clamped_y2 - clamped_y1));
}
