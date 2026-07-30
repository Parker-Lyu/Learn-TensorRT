#include "postprocess.hpp"
#include "preprocess.hpp"

#include <opencv2/core.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

bool near(float a, float b, float tolerance = 1.0e-4F) {
    return std::abs(a - b) <= tolerance;
}

void test_letterbox_wide_image() {
    const cv::Mat image(100, 200, CV_8UC3, cv::Scalar(10, 20, 30));
    lesson11::LetterboxInfo info;
    const cv::Mat letterboxed = lesson11::letterbox_image(image, cv::Size(640, 640), info);
    assert(letterboxed.cols == 640);
    assert(letterboxed.rows == 640);
    assert(info.resized_width == 640);
    assert(info.resized_height == 320);
    assert(info.pad_top == 160);
    assert(info.pad_bottom == 160);
    assert(near(info.scale, 3.2F));
}

void test_invalid_preprocess_inputs() {
    bool threw = false;
    try {
        lesson11::LetterboxInfo info;
        (void)lesson11::letterbox_image(cv::Mat{}, cv::Size(640, 640), info);
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw);
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
    assert(detections.size() == 2);
    assert(detections[0].class_id == 16);
    assert(detections[1].class_id == 0);
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
    assert(near(mapped.x1, 0.0F));
    assert(near(mapped.y1, 0.0F));
    assert(near(mapped.x2, 500.0F));
    assert(near(mapped.y2, 544.0F));
}

}  // namespace

int main() {
    test_letterbox_wide_image();
    test_invalid_preprocess_inputs();
    test_iou_and_nms_decode();
    test_map_box_clamps_padding();
    std::cout << "preprocess/postprocess tests passed\n";
    return 0;
}
