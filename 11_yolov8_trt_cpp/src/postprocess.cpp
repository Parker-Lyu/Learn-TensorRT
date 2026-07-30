#include "postprocess.hpp"

#include "preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace lesson11 {
namespace {

Box xywh_to_xyxy(float cx, float cy, float w, float h) {
    return Box{cx - 0.5F * w, cy - 0.5F * h, cx + 0.5F * w, cy + 0.5F * h};
}

float area(const Box& box) {
    return std::max(0.0F, box.x2 - box.x1) * std::max(0.0F, box.y2 - box.y1);
}

std::vector<int> nms(const std::vector<Box>& boxes,
                     const std::vector<float>& scores,
                     const std::vector<int>& indices,
                     float iou_threshold) {
    std::vector<int> order = indices;
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return scores[lhs] > scores[rhs];
    });

    std::vector<int> keep;
    while (!order.empty()) {
        const int current = order.front();
        keep.push_back(current);
        order.erase(order.begin());
        order.erase(std::remove_if(order.begin(), order.end(), [&](int candidate) {
                        return intersection_over_union(boxes[current], boxes[candidate]) >
                               iou_threshold;
                    }),
                    order.end());
    }
    return keep;
}

}  // namespace

const std::vector<std::string>& coco_class_names() {
    static const std::vector<std::string> names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"};
    return names;
}

float intersection_over_union(const Box& a, const Box& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float intersection = area(Box{x1, y1, x2, y2});
    const float union_area = area(a) + area(b) - intersection;
    return union_area <= 0.0F ? 0.0F : intersection / union_area;
}

std::vector<Detection> decode_yolov8_output(const std::vector<float>& output,
                                            const std::vector<int64_t>& output_shape,
                                            const LetterboxInfo& letterbox,
                                            const PostprocessConfig& config) {
    if (config.confidence_threshold < 0.0F || config.confidence_threshold > 1.0F ||
        config.iou_threshold < 0.0F || config.iou_threshold > 1.0F ||
        config.max_detections <= 0) {
        throw std::runtime_error("Invalid postprocess thresholds.");
    }
    if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] < 5) {
        throw std::runtime_error("Expected YOLO output shape [1, attributes, boxes].");
    }

    const int attributes = static_cast<int>(output_shape[1]);
    const int box_count = static_cast<int>(output_shape[2]);
    const int class_count = attributes - 4;
    if (output.size() != static_cast<std::size_t>(attributes) * box_count) {
        throw std::runtime_error("YOLO output size does not match output shape.");
    }

    std::vector<Box> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    boxes.reserve(box_count);
    scores.reserve(box_count);
    class_ids.reserve(box_count);

    for (int i = 0; i < box_count; ++i) {
        int best_class = 0;
        float best_score = output[(4 * box_count) + i];
        for (int cls = 1; cls < class_count; ++cls) {
            const float score = output[((4 + cls) * box_count) + i];
            if (score > best_score) {
                best_score = score;
                best_class = cls;
            }
        }
        if (best_score < config.confidence_threshold) {
            continue;
        }

        const float cx = output[(0 * box_count) + i];
        const float cy = output[(1 * box_count) + i];
        const float w = output[(2 * box_count) + i];
        const float h = output[(3 * box_count) + i];
        boxes.push_back(map_box_to_original(xywh_to_xyxy(cx, cy, w, h), letterbox));
        scores.push_back(best_score);
        class_ids.push_back(best_class);
    }

    std::vector<int> kept;
    for (int cls = 0; cls < class_count; ++cls) {
        std::vector<int> class_indices;
        for (int i = 0; i < static_cast<int>(class_ids.size()); ++i) {
            if (class_ids[i] == cls) {
                class_indices.push_back(i);
            }
        }
        const std::vector<int> class_kept = nms(boxes, scores, class_indices, config.iou_threshold);
        kept.insert(kept.end(), class_kept.begin(), class_kept.end());
    }

    std::sort(kept.begin(), kept.end(), [&](int lhs, int rhs) { return scores[lhs] > scores[rhs]; });
    if (static_cast<int>(kept.size()) > config.max_detections) {
        kept.resize(static_cast<std::size_t>(config.max_detections));
    }

    const std::vector<std::string>& names = coco_class_names();
    std::vector<Detection> detections;
    detections.reserve(kept.size());
    for (int index : kept) {
        Detection detection;
        detection.class_id = class_ids[index];
        detection.class_name = detection.class_id >= 0 &&
                                       detection.class_id < static_cast<int>(names.size())
                                   ? names[static_cast<std::size_t>(detection.class_id)]
                                   : std::to_string(detection.class_id);
        detection.confidence = scores[index];
        detection.box = boxes[index];
        detections.push_back(std::move(detection));
    }
    return detections;
}

}  // namespace lesson11
