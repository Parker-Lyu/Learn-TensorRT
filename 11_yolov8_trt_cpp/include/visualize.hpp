#pragma once

#include "yolo_types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace lesson11 {

cv::Mat draw_detections(const cv::Mat& bgr_image, const std::vector<Detection>& detections);

}  // namespace lesson11
