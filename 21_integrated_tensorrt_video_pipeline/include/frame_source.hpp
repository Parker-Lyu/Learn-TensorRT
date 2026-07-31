#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace lesson21 {

// FrameSource performs one incremental decode/read per call. The scheduler timestamps a frame only
// after read() succeeds, so capture-to-result latency includes video decode time.
class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual bool read(cv::Mat& frame) = 0;
    virtual std::string name() const = 0;
};

std::unique_ptr<FrameSource> make_repeatable_image_source(
    cv::Mat image, std::size_t frame_count, std::string name = "image");
std::unique_ptr<FrameSource> make_image_sequence_source(
    std::vector<cv::Mat> images, std::size_t frame_count, std::string name = "image-sequence");
std::unique_ptr<FrameSource> make_synthetic_source(
    cv::Size size, std::size_t frame_count, cv::Scalar color = {114, 114, 114});
std::unique_ptr<FrameSource> make_path_source(
    const std::string& path, std::size_t frame_count, bool repeat_video = false);

}  // namespace lesson21
