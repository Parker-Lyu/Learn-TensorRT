#pragma once

#include "yolo_types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace lesson10 {

struct PreprocessResult {
    std::vector<float> tensor_nchw;
    LetterboxInfo letterbox;
};

cv::Mat letterbox_image(const cv::Mat& bgr_image, cv::Size input_size, LetterboxInfo& info);

PreprocessResult preprocess_image(const cv::Mat& bgr_image, cv::Size input_size);

Box map_box_to_original(const Box& box_in_letterbox, const LetterboxInfo& info);

}  // namespace lesson10
