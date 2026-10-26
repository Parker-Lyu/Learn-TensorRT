#pragma once

#include "geometry.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace lesson11 {

struct PreprocessResult {
    std::vector<float> tensor_nchw;
    LetterboxInfo letterbox;
};

cv::Mat letterbox_image(const cv::Mat& bgr_image, cv::Size input_size, LetterboxInfo& info);

PreprocessResult preprocess_image(const cv::Mat& bgr_image, cv::Size input_size);


}  // namespace lesson11
