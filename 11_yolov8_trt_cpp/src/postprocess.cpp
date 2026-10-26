#include "postprocess.hpp"

#include "geometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace lesson11 {
namespace {

Box xywh_to_xyxy(float cx, float cy, float w, float h) {
    return Box{cx - 0.5F * w, cy - 0.5F * h, cx + 0.5F * w, cy + 0.5F * h};
}

double area(const Box& box) {
    return std::max(0.0, static_cast<double>(box.x2) - box.x1) *
           std::max(0.0, static_cast<double>(box.y2) - box.y1);
}

void validate_box(const Box& box) {
    if (!std::isfinite(box.x1) || !std::isfinite(box.y1) ||
        !std::isfinite(box.x2) || !std::isfinite(box.y2)) {
        throw std::runtime_error("Box coordinates must be finite.");
    }
}

std::vector<int> nms(const std::vector<Box>& boxes,
                     const std::vector<float>& scores,
                     const std::vector<int>& indices,
                     float iou_threshold,
                     int max_detections) {
    std::vector<int> order = indices;
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return scores[lhs] > scores[rhs];
    });

    std::vector<int> keep;
    keep.reserve(order.size());
    for (int candidate : order) {
        const bool suppressed = std::any_of(keep.begin(), keep.end(), [&](int selected) {
            return intersection_over_union(boxes[selected], boxes[candidate]) > iou_threshold;
        });
        if (!suppressed) {
            keep.push_back(candidate);
            // No class can contribute more than the final global limit.
            if (keep.size() == static_cast<std::size_t>(max_detections)) {
                break;
            }
        }
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
    validate_box(a);
    validate_box(b);
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const double intersection = area(Box{x1, y1, x2, y2});
    const double union_area = area(a) + area(b) - intersection;
    return union_area <= 0.0 ? 0.0F : static_cast<float>(intersection / union_area);
}

std::vector<Detection> decode_yolov8_output(const std::vector<float>& output,
                                            const std::vector<int64_t>& output_shape,
                                            const LetterboxInfo& letterbox,
                                            const PostprocessConfig& config) {
    validate_letterbox_info(letterbox);
    if (!std::isfinite(config.confidence_threshold) || !std::isfinite(config.iou_threshold) ||
        config.confidence_threshold < 0.0F || config.confidence_threshold > 1.0F ||
        config.iou_threshold < 0.0F || config.iou_threshold > 1.0F ||
        config.max_detections <= 0) {
        throw std::runtime_error("Invalid postprocess thresholds.");
    }
    if (output_shape.size() != 3 || output_shape[0] != 1 || output_shape[1] < 5 ||
        output_shape[1] > std::numeric_limits<int>::max() || output_shape[2] < 0 ||
        output_shape[2] > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Expected YOLO output shape [1, attributes, boxes].");
    }

    const int attributes = static_cast<int>(output_shape[1]);
    const std::size_t box_count = static_cast<std::size_t>(output_shape[2]);
    const int class_count = attributes - 4;
    if (box_count > std::numeric_limits<std::size_t>::max() /
                        static_cast<std::size_t>(attributes) ||
        output.size() != static_cast<std::size_t>(attributes) * box_count) {
        throw std::runtime_error("YOLO output size does not match output shape.");
    }

    if (!std::all_of(output.begin(), output.end(), [](float value) {
            return std::isfinite(value);
        })) {
        throw std::runtime_error("YOLO output must contain only finite values.");
    }

    std::vector<Box> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    boxes.reserve(box_count);
    scores.reserve(box_count);
    class_ids.reserve(box_count);

    for (std::size_t i = 0; i < box_count; ++i) {
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
        if (w <= 0.0F || h <= 0.0F) {
            continue;
        }
        const Box mapped = map_box_to_original(xywh_to_xyxy(cx, cy, w, h), letterbox);
        if (mapped.x2 <= mapped.x1 || mapped.y2 <= mapped.y1) {
            continue;  // A box wholly in padding is not a detection in the source image.
        }
        boxes.push_back(mapped);
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
        const std::vector<int> class_kept = nms(boxes, scores, class_indices, config.iou_threshold,
                                                  config.max_detections);
        kept.insert(kept.end(), class_kept.begin(), class_kept.end());
    }

    std::stable_sort(kept.begin(), kept.end(), [&](int lhs, int rhs) { return scores[lhs] > scores[rhs]; });
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
