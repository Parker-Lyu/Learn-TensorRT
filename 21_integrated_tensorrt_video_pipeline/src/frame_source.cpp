#include "frame_source.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include <stdexcept>
#include <sstream>
#include <utility>

namespace lesson21 {
namespace {

class ImageSequenceSource final : public FrameSource {
public:
    ImageSequenceSource(std::vector<cv::Mat> images, std::size_t frame_count, std::string source_name)
        : images_(std::move(images)), frame_count_(frame_count), name_(std::move(source_name)) {
        if (images_.empty() || frame_count_ == 0) {
            throw std::invalid_argument("image source and frame count must be non-empty");
        }
        for (const cv::Mat& image : images_) {
            if (image.empty()) throw std::invalid_argument("image source contains an empty frame");
        }
    }

    bool read(cv::Mat& frame) override {
        if (next_ == frame_count_) return false;
        // cv::Mat reference counting is safe here because the source retains the immutable image.
        frame = images_[next_ % images_.size()];
        ++next_;
        return true;
    }

    std::string name() const override { return name_; }

private:
    std::vector<cv::Mat> images_;
    std::size_t frame_count_;
    std::size_t next_{0};
    std::string name_;
};

class VideoFileSource final : public FrameSource {
public:
    VideoFileSource(std::string path, std::size_t frame_count, bool repeat)
        : path_(std::move(path)), frame_count_(frame_count), repeat_(repeat), capture_(path_) {
        if (frame_count_ == 0) throw std::invalid_argument("video frame count must be positive");
        if (!capture_.isOpened()) throw std::runtime_error("cannot open video source: " + path_);
    }

    bool read(cv::Mat& frame) override {
        if (next_ == frame_count_) return false;
        if (!capture_.read(frame)) {
            if (!repeat_) return false;
            capture_.release();
            capture_.open(path_);
            if (!capture_.isOpened() || !capture_.read(frame)) {
                throw std::runtime_error("cannot restart video source: " + path_);
            }
        }
        if (frame.empty()) throw std::runtime_error("video decoder returned an empty frame: " + path_);
        ++next_;
        return true;
    }

    std::string name() const override { return path_; }

private:
    std::string path_;
    std::size_t frame_count_;
    bool repeat_;
    std::size_t next_{0};
    cv::VideoCapture capture_;
};

}  // namespace

std::unique_ptr<FrameSource> make_repeatable_image_source(
        cv::Mat image, std::size_t frame_count, std::string name) {
    return make_image_sequence_source({std::move(image)}, frame_count, std::move(name));
}

std::unique_ptr<FrameSource> make_image_sequence_source(
        std::vector<cv::Mat> images, std::size_t frame_count, std::string name) {
    return std::make_unique<ImageSequenceSource>(
        std::move(images), frame_count, std::move(name));
}

std::unique_ptr<FrameSource> make_synthetic_source(
        cv::Size size, std::size_t frame_count, cv::Scalar color) {
    if (size.width <= 0 || size.height <= 0) {
        throw std::invalid_argument("synthetic source size must be positive");
    }
    return make_repeatable_image_source(
        cv::Mat(size, CV_8UC3, color), frame_count, "synthetic");
}

std::unique_ptr<FrameSource> make_path_source(
        const std::string& path, std::size_t frame_count, bool repeat_video) {
    if (path == "synthetic") {
        return make_synthetic_source({640, 480}, frame_count);
    }
    constexpr const char* sequence_prefix = "sequence:";
    if (path.rfind(sequence_prefix, 0) == 0) {
        std::vector<cv::Mat> images;
        std::stringstream paths(path.substr(std::char_traits<char>::length(sequence_prefix)));
        std::string item;
        while (std::getline(paths, item, '|')) {
            cv::Mat sequence_image = cv::imread(item, cv::IMREAD_COLOR);
            if (sequence_image.empty()) {
                throw std::runtime_error("cannot read image sequence item: " + item);
            }
            images.push_back(std::move(sequence_image));
        }
        return make_image_sequence_source(std::move(images), frame_count, path);
    }
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (!image.empty()) {
        return make_repeatable_image_source(std::move(image), frame_count, path);
    }
    return std::make_unique<VideoFileSource>(path, frame_count, repeat_video);
}

}  // namespace lesson21
