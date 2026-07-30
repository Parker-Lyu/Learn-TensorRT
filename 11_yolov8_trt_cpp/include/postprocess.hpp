#pragma once

#include "yolo_types.hpp"

#include <vector>

namespace lesson11 {

struct PostprocessConfig {
    float confidence_threshold = 0.25F;
    float iou_threshold = 0.45F;
    int max_detections = 100;
};

const std::vector<std::string>& coco_class_names();

float intersection_over_union(const Box& a, const Box& b);

std::vector<Detection> decode_yolov8_output(const std::vector<float>& output,
                                            const std::vector<int64_t>& output_shape,
                                            const LetterboxInfo& letterbox,
                                            const PostprocessConfig& config);

}  // namespace lesson11
