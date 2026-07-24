#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>

namespace {

constexpr const char* kDefaultImagePath = "../assets/img.jpeg";

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [image_path]\n";
}

struct DepthDescription {
    const char* opencv_name;
    const char* element_name;
};

DepthDescription describe_depth(int depth) {
    switch (depth) {
        case CV_8U:
            return {"CV_8U", "uint8"};
        case CV_8S:
            return {"CV_8S", "int8"};
        case CV_16U:
            return {"CV_16U", "uint16"};
        case CV_16S:
            return {"CV_16S", "int16"};
        case CV_32S:
            return {"CV_32S", "int32"};
        case CV_32F:
            return {"CV_32F", "float32"};
        case CV_64F:
            return {"CV_64F", "float64"};
        case CV_16F:
            return {"CV_16F", "float16"};
        default:
            return {"unknown", "unknown"};
    }
}

std::string opencv_type_name(const cv::Mat& image) {
    const DepthDescription depth = describe_depth(image.depth());
    return std::string(depth.opencv_name) + "C" + std::to_string(image.channels());
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_usage(argv[0]);
        return 0;
    }

    if (argc > 2 || (argc == 2 && argv[1][0] == '-')) {
        std::cerr << "Invalid command-line arguments.\n";
        print_usage(argv[0]);
        return 2;
    }

    // With no explicit path, use the repository's shared sample image. The
    // README runs the executable from this lesson directory, making this path
    // independent of the build-directory name.
    const std::string image_path = argc == 2 ? argv[1] : kDefaultImagePath;

    // cv::imread loads an image from disk into a cv::Mat object.
    // IMREAD_UNCHANGED preserves the file's decoded depth and channel count so
    // the reported metadata describes the decoded source rather than a forced
    // three-channel conversion.
    cv::Mat image;
    try {
        image = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    } catch (const cv::Exception& error) {
        std::cerr << "OpenCV could not read the image: " << error.what() << '\n';
        return 1;
    }

    // An empty cv::Mat means the image could not be loaded.
    if (image.empty()) {
        std::cerr << "Failed to read image: " << image_path << '\n';
        return 1;
    }

    std::cout << "Loaded image: " << image_path << '\n';
    std::cout << "Width: " << image.cols << '\n';
    std::cout << "Height: " << image.rows << '\n';
    std::cout << "Channels: " << image.channels() << '\n';
    const DepthDescription depth = describe_depth(image.depth());
    std::cout << "Data type: " << opencv_type_name(image) << " ("
              << depth.element_name << " per channel)\n";

    return 0;
}
