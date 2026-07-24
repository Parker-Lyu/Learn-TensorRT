#include "preprocess.hpp"

#include <opencv2/core.hpp>

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_near(float actual, float expected, float tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                                 ", expected=" + std::to_string(expected));
    }
}

void expect_throws(const std::function<void()>& operation, const std::string& message) {
    try {
        operation();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void test_letterbox_landscape_and_odd_padding() {
    const cv::Mat image(2, 3, CV_8UC3, cv::Scalar(1, 2, 3));
    LetterboxInfo info;
    const cv::Mat output = letterbox_image(image, cv::Size(8, 8), info);

    expect(output.cols == 8 && output.rows == 8 && output.type() == CV_8UC3,
           "letterbox output shape or type is wrong");
    expect(info.resized_width == 8 && info.resized_height == 5,
           "aspect-ratio resize dimensions are wrong");
    expect(info.pad_left == 0 && info.pad_right == 0 && info.pad_top == 1 && info.pad_bottom == 2,
           "letterbox padding is wrong");
    expect(output.at<cv::Vec3b>(0, 0) == cv::Vec3b(114, 114, 114),
           "padding color is wrong");
}

void test_extreme_portrait_shape() {
    const cv::Mat image(100, 1, CV_8UC3, cv::Scalar(0, 0, 0));
    LetterboxInfo info;
    letterbox_image(image, cv::Size(640, 640), info);
    expect(info.resized_width == 6 && info.resized_height == 640,
           "extreme portrait resize is wrong");
    expect(info.pad_left == 317 && info.pad_right == 317,
           "extreme portrait padding is wrong");
}

void test_rgb_normalization_and_batch_layout() {
    const cv::Mat first(1, 1, CV_8UC3, cv::Scalar(10, 20, 30));
    const cv::Mat second(1, 1, CV_8UC3, cv::Scalar(40, 50, 60));
    const BatchPreprocessResult result =
        preprocess_batch_to_nchw(std::vector<cv::Mat>{first, second}, cv::Size(1, 1));

    expect(result.batch_size == 2 && result.channels == 3 && result.height == 1 &&
               result.width == 1,
           "batch tensor metadata is wrong");
    expect(result.input_tensor.size() == 6, "batch tensor size is wrong");
    expect_near(result.input_tensor[0], 30.0F / 255.0F, 1.0e-6F, "first red value is wrong");
    expect_near(result.input_tensor[1], 20.0F / 255.0F, 1.0e-6F, "first green value is wrong");
    expect_near(result.input_tensor[2], 10.0F / 255.0F, 1.0e-6F, "first blue value is wrong");
    expect_near(result.input_tensor[3], 60.0F / 255.0F, 1.0e-6F, "second red value is wrong");
}

void test_coordinate_mapping_and_clamping() {
    const cv::Mat image(100, 200, CV_8UC3, cv::Scalar(0, 0, 0));
    LetterboxInfo info;
    letterbox_image(image, cv::Size(400, 400), info);

    const cv::Rect2f mapped = map_box_to_original_image(cv::Rect2f(40, 80, 400, 280), info);
    expect_near(mapped.x, 20.0F, 1.0e-6F, "mapped x is wrong");
    expect_near(mapped.y, 0.0F, 1.0e-6F, "mapped y clamp is wrong");
    expect_near(mapped.width, 180.0F, 1.0e-6F, "mapped width clamp is wrong");
    expect_near(mapped.height, 100.0F, 1.0e-6F, "mapped height clamp is wrong");
}

void test_invalid_inputs() {
    LetterboxInfo info;
    expect_throws([&] { letterbox_image(cv::Mat(), cv::Size(640, 640), info); },
                  "empty image should fail");
    expect_throws(
        [&] { letterbox_image(cv::Mat(2, 2, CV_8UC1), cv::Size(640, 640), info); },
        "grayscale image should fail");
    expect_throws(
        [&] { letterbox_image(cv::Mat(2, 2, CV_32FC3), cv::Size(640, 640), info); },
        "non-uint8 image should fail");
    expect_throws(
        [&] { letterbox_image(cv::Mat(2, 2, CV_8UC3), cv::Size(0, 640), info); },
        "non-positive target size should fail");
    expect_throws([&] { preprocess_batch_to_nchw({}, cv::Size(640, 640)); },
                  "empty batch should fail");
    expect_throws([&] { map_box_to_original_image(cv::Rect2f(0, 0, -1, 2), info); },
                  "invalid LetterboxInfo should fail");

    letterbox_image(cv::Mat(2, 2, CV_8UC3), cv::Size(640, 640), info);
    expect_throws([&] { map_box_to_original_image(cv::Rect2f(0, 0, -1, 2), info); },
                  "negative box extent should fail");
}

}  // namespace

int main() {
    try {
        test_letterbox_landscape_and_odd_padding();
        test_extreme_portrait_shape();
        test_rgb_normalization_and_batch_layout();
        test_coordinate_mapping_and_clamping();
        test_invalid_inputs();
        std::cout << "All preprocessing tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
}
