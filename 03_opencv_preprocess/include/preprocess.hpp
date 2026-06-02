#pragma once

#include <opencv2/core.hpp>

#include <vector>

struct LetterboxInfo {
    int original_width = 0;
    int original_height = 0;
    int input_width = 0;
    int input_height = 0;
    int resized_width = 0;
    int resized_height = 0;
    int pad_left = 0;
    int pad_top = 0;
    int pad_right = 0;
    int pad_bottom = 0;
    float scale = 1.0F;
};

struct BatchPreprocessResult {
    std::vector<float> input_tensor;
    std::vector<LetterboxInfo> letterbox_infos;
    int batch_size = 0;
    int channels = 3;
    int height = 0;
    int width = 0;
};

cv::Mat letterbox_image(const cv::Mat& bgr_image, cv::Size input_size, LetterboxInfo& info);

BatchPreprocessResult preprocess_batch_to_nchw(const std::vector<cv::Mat>& bgr_images,
                                               cv::Size input_size);

cv::Rect2f map_box_to_original_image(const cv::Rect2f& box_in_letterbox,
                                     const LetterboxInfo& info);
