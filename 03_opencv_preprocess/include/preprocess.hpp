#pragma once

#include <opencv2/core.hpp>

#include <vector>

struct LetterboxInfo {
    // Original image size before any resize or padding. Detection boxes must end up here.
    int original_width = 0;
    int original_height = 0;

    // Network input size after letterbox. This is the H x W expected by the engine.
    int input_width = 0;
    int input_height = 0;

    // Size of the aspect-ratio-preserving resize before padding is added.
    int resized_width = 0;
    int resized_height = 0;

    // Padding in network-input pixels. Keep all four values because odd padding is asymmetric.
    int pad_left = 0;
    int pad_top = 0;
    int pad_right = 0;
    int pad_bottom = 0;

    // Multiplier from original image coordinates into the resized image.
    float scale = 1.0F;
};

struct BatchPreprocessResult {
    // Contiguous tensor buffer in NCHW order: n * C * H * W + c * H * W + y * W + x.
    std::vector<float> input_tensor;

    // One letterbox record per image, needed later to map model boxes back to each source image.
    std::vector<LetterboxInfo> letterbox_infos;

    int batch_size = 0;
    int channels = 3;
    int height = 0;
    int width = 0;
};

// The image contract is deliberately explicit: uint8 BGR HWC in, as returned by
// cv::imread(..., cv::IMREAD_COLOR).
cv::Mat letterbox_image(const cv::Mat& bgr_image, cv::Size input_size, LetterboxInfo& info);

BatchPreprocessResult preprocess_batch_to_nchw(const std::vector<cv::Mat>& bgr_images,
                                               cv::Size input_size);

cv::Rect2f map_box_to_original_image(const cv::Rect2f& box_in_letterbox,
                                     const LetterboxInfo& info);
