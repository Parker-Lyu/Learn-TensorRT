#include "preprocess.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition) {
    if (!condition) {
        throw std::runtime_error("preprocess/postprocess check failed");
    }
}

template <typename Function>
void require_throws(Function function) {
    bool threw = false;
    try {
        function();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    require(threw);
}


bool near(float a, float b, float tolerance = 1.0e-4F) {
    return std::abs(a - b) <= tolerance;
}

void test_letterbox_wide_image() {
    const cv::Mat image(100, 200, CV_8UC3, cv::Scalar(10, 20, 30));
    lesson11::LetterboxInfo info;
    const cv::Mat letterboxed = lesson11::letterbox_image(image, cv::Size(640, 640), info);
    require(letterboxed.cols == 640);
    require(letterboxed.rows == 640);
    require(info.resized_width == 640);
    require(info.resized_height == 320);
    require(info.pad_top == 160);
    require(info.pad_bottom == 160);
    require(near(info.scale, 3.2F));
}

void test_invalid_preprocess_inputs() {
    bool threw = false;
    try {
        lesson11::LetterboxInfo info;
        (void)lesson11::letterbox_image(cv::Mat{}, cv::Size(640, 640), info);
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw);
    lesson11::LetterboxInfo info;
    require_throws([&] {
        lesson11::letterbox_image(cv::Mat(2, 2, CV_32FC3), cv::Size(640, 640), info);
    });
}

void test_tall_odd_padding_and_rgb() {
    lesson11::LetterboxInfo info;
    const cv::Mat tall(200, 101, CV_8UC3, cv::Scalar(0, 127, 255));
    lesson11::letterbox_image(tall, cv::Size(640, 640), info);
    require(info.resized_width == 323 && info.pad_left == 158 && info.pad_right == 159);
    require(info.resized_height == 640 && info.pad_top == 0);
    // A non-contiguous ROI exercises row-stride-aware planar conversion.
    const cv::Mat source(2, 3, CV_8UC3, cv::Scalar(0, 127, 255));
    const auto result = lesson11::preprocess_image(source(cv::Rect(0, 0, 2, 2)), cv::Size(2, 2));
    require(result.tensor_nchw.size() == 12);
    for (int i = 0; i < 4; ++i) {
        require(near(result.tensor_nchw[i], 1.0F));
        require(near(result.tensor_nchw[4 + i], 127.0F / 255.0F));
        require(near(result.tensor_nchw[8 + i], 0.0F));
    }
}

}  // namespace

int main() {
    test_letterbox_wide_image();
    test_invalid_preprocess_inputs();
    test_tall_odd_padding_and_rgb();
    std::cout << "preprocess/postprocess tests passed\n";
    return 0;
}
