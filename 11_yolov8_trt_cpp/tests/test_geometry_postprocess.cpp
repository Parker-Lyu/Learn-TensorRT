#include "postprocess.hpp"
#include "geometry.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("geometry/postprocess check failed");
    }
}
template <typename Function>
void require_throws(Function function) {
    try {
        function();
    } catch (const std::runtime_error&) {
        return;
    }
    throw std::runtime_error("expected geometry rejection");
}
bool near(float a, float b, float tolerance = 1.0e-4F) {
    return std::abs(a - b) <= tolerance;
}

void test_iou_and_nms_decode() {
    lesson11::LetterboxInfo info;
    info.original_width = 640;
    info.original_height = 640;
    info.input_width = 640;
    info.input_height = 640;
    info.scale = 1.0F;

    const int attributes = 4 + 80;
    const int boxes = 3;
    std::vector<float> output(static_cast<std::size_t>(attributes) * boxes, 0.0F);
    auto set_box = [&](int index, float cx, float cy, float w, float h, int cls, float score) {
        output[0 * boxes + index] = cx;
        output[1 * boxes + index] = cy;
        output[2 * boxes + index] = w;
        output[3 * boxes + index] = h;
        output[(4 + cls) * boxes + index] = score;
    };
    set_box(0, 100.0F, 100.0F, 50.0F, 50.0F, 16, 0.90F);
    set_box(1, 102.0F, 102.0F, 50.0F, 50.0F, 16, 0.80F);
    set_box(2, 300.0F, 300.0F, 40.0F, 40.0F, 0, 0.70F);

    lesson11::PostprocessConfig config;
    config.confidence_threshold = 0.25F;
    config.iou_threshold = 0.45F;
    const std::vector<lesson11::Detection> detections =
        lesson11::decode_yolov8_output(output, {1, attributes, boxes}, info, config);
    require(detections.size() == 2);
    require(detections[0].class_id == 16);
    require(detections[1].class_id == 0);
}

void test_map_box_clamps_padding() {
    lesson11::LetterboxInfo info;
    info.original_width = 500;
    info.original_height = 544;
    info.input_width = 640;
    info.input_height = 640;
    info.pad_left = 26;
    info.pad_top = 0;
    info.scale = 1.1764706F;

    const lesson11::Box mapped =
        lesson11::map_box_to_original(lesson11::Box{0.0F, -5.0F, 700.0F, 700.0F}, info);
    require(near(mapped.x1, 0.0F));
    require(near(mapped.y1, 0.0F));
    require(near(mapped.x2, 500.0F));
    require(near(mapped.y2, 544.0F));
}

void test_defensive_geometry_and_decode() {
    lesson11::LetterboxInfo info;
    info.original_width = info.original_height = 640;
    info.scale = 0.0F;
    require_throws([&] { lesson11::map_box_to_original({}, info); });
    info.scale = 1.0F;
    const lesson11::PostprocessConfig config;
    require_throws([&] { lesson11::decode_yolov8_output({}, {1, 5, -1}, info, config); });
    require_throws([&] {
        lesson11::decode_yolov8_output({}, {1, 5, std::numeric_limits<int64_t>::max()}, info, config);
    });
    auto invalid_config = config;
    invalid_config.iou_threshold = std::numeric_limits<float>::quiet_NaN();
    require_throws([&] { lesson11::decode_yolov8_output({}, {1, 5, 0}, info, invalid_config); });
    require(lesson11::decode_yolov8_output({}, {1, 5, 0}, info, config).empty());

    // Two overlapping, equal-score boxes retain the first input deterministically.
    std::vector<float> output = {100, 101, 100, 100, 50, 50, 50, 50, 0.9F, 0.9F};
    auto detections = lesson11::decode_yolov8_output(output, {1, 5, 2}, info, config);
    require(detections.size() == 1 && near(detections[0].box.x1, 75.0F));
    output[8] = std::numeric_limits<float>::quiet_NaN();
    require_throws([&] { lesson11::decode_yolov8_output(output, {1, 5, 2}, info, config); });
}

void test_iou_extremes() {
    using lesson11::intersection_over_union;
    const float large = std::numeric_limits<float>::max();
    const lesson11::Box full{-large, -large, large, large};
    require(near(intersection_over_union(full, full), 1.0F));
    require(intersection_over_union({1, 1, 0, 0}, full) == 0.0F);
    require(intersection_over_union({}, full) == 0.0F);
    require(intersection_over_union({0, 0, 1, 1}, {2, 2, 3, 3}) == 0.0F);
    require(near(intersection_over_union({0, 0, 2, 2}, {1, 1, 3, 3}), 1.0F / 7.0F));
    require_throws([&] {
        intersection_over_union({0, 0, std::numeric_limits<float>::quiet_NaN(), 1}, full);
    });
}

void test_padding_and_empty_metadata() {
    lesson11::LetterboxInfo info;
    const lesson11::PostprocessConfig config;
    require_throws([&] { lesson11::decode_yolov8_output({}, {1, 5, 0}, info, config); });
    info.original_width = info.original_height = 100;
    info.pad_left = 100;
    // Positive-area model-space box wholly in left padding must not survive clipping.
    require(lesson11::decode_yolov8_output({10, 10, 5, 5, 0.9F}, {1, 5, 1}, info, config).empty());
    info.pad_left = 0;
    info.scale = std::numeric_limits<float>::denorm_min();
    const auto mapped = lesson11::map_box_to_original({-1, -1, 1, 1}, info);
    require(mapped.x1 == 0 && mapped.x2 == 100 && mapped.y1 == 0 && mapped.y2 == 100);
}

void test_classwise_threshold_and_cap() {
    lesson11::LetterboxInfo info;
    info.original_width = info.original_height = 1000;
    lesson11::PostprocessConfig config;
    std::vector<float> same = {100, 100, 100, 100, 50, 50, 50, 50, 0.9F, 0.9F};
    config.iou_threshold = 1.0F;
    require(lesson11::decode_yolov8_output(same, {1, 5, 2}, info, config).size() == 2);
    config.iou_threshold = 0.0F;
    require(lesson11::decode_yolov8_output(same, {1, 5, 2}, info, config).size() == 1);
    same = {100, 100, 100, 100, 50, 50, 50, 50, 0.9F, 0.0F, 0.0F, 0.9F};
    require(lesson11::decode_yolov8_output(same, {1, 6, 2}, info, config).size() == 2);
    constexpr int count = 10;
    std::vector<float> output(6 * count, 0.0F);
    for (int i = 0; i < count; ++i) {
        output[i] = 20.0F + 30.0F * i;
        output[count + i] = 100.0F;
        output[2 * count + i] = output[3 * count + i] = 10.0F;
        output[(4 + i % 2) * count + i] = 0.9F;  // Include cross-class score ties.
    }
    const auto all = lesson11::decode_yolov8_output(output, {1, 6, count}, info, config);
    config.max_detections = 3;
    const auto capped = lesson11::decode_yolov8_output(output, {1, 6, count}, info, config);
    require(all.size() == count && capped.size() == 3);
    for (std::size_t i = 0; i < capped.size(); ++i) {
        require(capped[i].class_id == all[i].class_id && capped[i].box.x1 == all[i].box.x1);
    }
}

}  // namespace
int main() {
    test_iou_and_nms_decode();
    test_map_box_clamps_padding();
    test_defensive_geometry_and_decode();
    test_iou_extremes();
    test_padding_and_empty_metadata();
    test_classwise_threshold_and_cap();
    std::cout << "geometry/postprocess tests passed\n";
}
